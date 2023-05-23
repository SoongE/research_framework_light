import logging

import torch
import torchmetrics
from timm.utils import dispatch_clip_grad, distribute_bn, update_summary
from torch.cuda.amp import autocast
from torchmetrics import MeanMetric
from torchmetrics.functional import accuracy


class Fit:
    def __init__(self, cfg, scaler, device, epochs, model, criterion, optimizer, model_ema, scheduler, saver,
                 loader=None):
        self.device = device
        self.local_rank = cfg.local_rank
        self.world_size = cfg.world_size

        self.distributed = cfg.distributed
        self.dist_bn = cfg.train.dist_bn
        self.clip_grad = cfg.train.optimizer.clip_grad
        self.clip_mode = cfg.train.optimizer.clip_mode
        self.grad_accumulation = cfg.train.optimizer.grad_accumulation
        self.double_valid = cfg.train.double_valid
        self.wandb = cfg.wandb
        self.start_epoch, self.num_epochs = epochs
        self.logging_interval = 100
        self.num_classes = cfg.dataset.num_classes
        self.tm = cfg.train.target_metric
        self.eval_metrics = cfg.train.eval_metrics

        self.model = model
        if isinstance(criterion, list):
            self.train_criterion = criterion[0]
            self.val_criterion = criterion[1]
        else:
            self.train_criterion = self.val_criterion = criterion
        self.optimizer = optimizer
        self.model_ema = model_ema
        self.scaler = scaler
        self.scheduler = scheduler
        self.saver = saver
        if loader:
            self.train_loader = loader[0]
            self.val_loader = loader[1]

        self.losses = MeanMetric(compute_on_step=False).to(self.device)
        self.metric_fn = self.init_metrics(cfg.dataset.task, 0.5, cfg.dataset.num_classes, cfg.dataset.num_classes,
                                           'micro')

    def __call__(self, *args, **kwargs):
        for epoch in range(self.start_epoch, self.num_epochs):
            if self.distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            train_metrics = self.train(epoch)
            if self.distributed and self.dist_bn in ('broadcast', 'reduce'):
                distribute_bn(self.model, self.world_size, self.dist_bn == 'reduce')

            if self.model_ema:
                if self.distributed and self.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(self.model_ema, self.world_size, self.dist_bn == 'reduce')
                eval_metrics = self.validate(epoch, ema=True)
                if self.double_valid:
                    _ = self.validate(epoch, ema=False)
            else:
                eval_metrics = self.validate(epoch, ema=False)

            # save proper checkpoint with eval metric
            if self._master_node():
                torch.cuda.synchronize()
                best_metric, best_epoch = self.saver.save_checkpoint(epoch, metric=eval_metrics[self.tm])
                eval_metrics.update({f'Best_{self.tm}': best_metric})
                update_summary(epoch, train_metrics, eval_metrics, 'summary.csv', log_wandb=self.wandb,
                               write_header=(epoch == 0))

        if self._master_node():
            logging.info(f'*** Best {self.tm}: {best_metric} {best_epoch} ***')

    def iterate(self, model, data, criterion):
        x, y = map(lambda x: x.to(self.device), data)
        x = x.to(memory_format=torch.channels_last)

        with autocast(enabled=True if self.scaler else False):
            prob = model(x)
            loss = criterion(prob, y)

        return loss, prob, y

    def train(self, epoch):
        self._reset_metric()
        total = len(self.train_loader)

        self.model.train()
        self.optimizer.zero_grad()
        for i, data in enumerate(self.train_loader):
            loss, prob, target = self.iterate(self.model, data, self.train_criterion)
            loss /= self.grad_accumulation

            self._backward(loss, (i + 1) % self.grad_accumulation == 0)

            self.losses.update(loss)
            if self._master_node() and i % self.logging_interval == 0:
                logging.info(f'Train {epoch:>3}: [{i:>4}/{total}]  loss:{self.losses.compute():.6f}')

            torch.cuda.synchronize()
            self.scheduler.step()

            if i == 2:
                break

        if hasattr(self.optimizer, 'sync_lookahead'):
            self.optimizer.sync_lookahead()

        return {'loss': self.losses.compute().item()}

    @torch.no_grad()
    def validate(self, epoch, ema=False):
        model = self.model_ema if ema else self.model
        log_prefix = ' EMA ' if ema else ' Val '
        self._reset_metric()
        total = len(self.val_loader)

        model.eval()
        for i, data in enumerate(self.val_loader):
            loss, prob, target = self.iterate(model, data, self.val_criterion)
            loss /= self.grad_accumulation
            self._update_metric(loss, prob, target)

            if self._master_node() and i % self.logging_interval == 0:
                logging.info(self._print(self._metrics(), epoch, i, total, log_prefix))

        return self._metrics()

    @torch.no_grad()
    def test(self, ema, test_loader):
        self._reset_metric()
        accuracies_list = list()
        total = len(test_loader)

        model = self.model_ema if ema else self.model

        model.eval()
        for i, data in enumerate(test_loader):
            loss, prob, target = self.iterate(model, data, self.val_criterion)
            self._update_metric(loss, prob, target)
            accuracies_list.append(accuracy(prob, target))

            if self._master_node() and i % self.logging_interval == 0:
                logging.info(self._print(self._metrics(), 0, i, total, 'Test'))

        return accuracies_list, self._metrics()

    def _backward(self, loss, update_grad):
        if self.scaler:
            self._scaler_backward(loss, update_grad)
        else:
            self._default_backward(loss, update_grad)

        if update_grad:
            self.optimizer.zero_grad()
            if self.model_ema:
                self.model_ema.update(self.model)

    def _scaler_backward(self, loss, update_grad):
        self.scaler(loss, self.optimizer, self.model.parameters(), self.clip_grad, update_grad)

    def _default_backward(self, loss, update_grad):
        loss.backward()
        if update_grad:
            if self.clip_grad:
                dispatch_clip_grad(self.model.parameters(), self.clip_grad, mode=self.clip_mode)
            self.optimizer.step()

    def _update_metric(self, loss, prob, target):
        self.losses.update(loss.item() / prob.size(0))
        for fn in self.metric_fn.values():
            fn.update(prob, target)

    def _reset_metric(self):
        self.losses.reset()
        for fn in self.metric_fn.values():
            fn.reset()

    def _metrics(self):
        result = dict()
        result['loss'] = self.losses.compute().tolist()
        for k, fn in self.metric_fn.items():
            result[k] = fn.compute().tolist()
        return result

    @staticmethod
    def _print(metrics, epoch, i, max_iter, mode):
        log = f'{mode} {epoch:>3}: [{i:>4d}/{max_iter}]  '
        if "ConfusionMatrix" in metrics:
            metrics.pop('ConfusionMatrix')
        for k, v in metrics.items():
            log += f'{k}:{v:.6f} | '
        return log[:-3]

    def _master_node(self):
        return self.local_rank == 0

    def init_metrics(self, task, threshold, num_class, num_label, average, top_k=1):
        metric_fn = dict()

        for metric in self.eval_metrics:
            metric_fn[metric] = torchmetrics.__dict__[metric](task=task, threshold=threshold, num_classes=num_class,
                                                              compute_on_step=False,
                                                              average='macro' if metric == 'AUROC' else average,
                                                              num_labels=num_label, top_k=top_k).to(self.device)
        return metric_fn
