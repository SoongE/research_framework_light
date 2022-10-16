import logging

import torch
from timm.models import model_parameters
from timm.utils import dispatch_clip_grad, distribute_bn, update_summary
from torch.cuda.amp import autocast
from torchmetrics import MeanMetric, Accuracy
from torchmetrics.functional import accuracy
from tqdm import tqdm


class Fit:
    def __init__(self, cfg, scaler, device, local_rank, start_epoch, num_epochs, model, criterion, optimizer, model_ema,
                 scheduler, saver, loader=None):
        self.device = device
        self.local_rank = local_rank
        self.world_size = cfg.world_size

        self.distributed = cfg.distributed
        self.dist_bn = cfg.train.dist_bn
        self.clip_grad = cfg.train.optimizer.clip_grad
        self.clip_mode = cfg.train.optimizer.clip_mode
        self.grad_accumulation = cfg.train.optimizer.grad_accumulation
        self.double_valid = cfg.train.double_valid
        self.wandb = cfg.wandb
        self.start_epoch = start_epoch
        self.num_epochs = num_epochs

        self.model = model
        self.train_criterion = criterion[0]
        self.val_criterion = criterion[1]
        self.optimizer = optimizer
        self.model_ema = model_ema
        self.scaler = scaler
        self.scheduler = scheduler
        self.saver = saver
        if loader:
            self.train_loader = loader[0]
            self.val_loader = loader[1]

        self.losses = MeanMetric(compute_on_step=False).to(self.device)
        self.top1 = Accuracy(compute_on_step=False).to(self.device)
        self.top5 = Accuracy(compute_on_step=False, top_k=5).to(self.device)

    def __call__(self, *args, **kwargs):
        for epoch in range(self.start_epoch, self.num_epochs):
            if self.distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            train_metrics = self.train(epoch)
            if self.distributed and self.dist_bn in ('broadcast', 'reduce'):
                print("DISTRIBUTED")
                distribute_bn(self.model, self.world_size, self.dist_bn == 'reduce')

            if self.model_ema:
                if self.distributed and self.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(self.model_ema, self.world_size, self.dist_bn == 'reduce')
                eval_metrics = self.validate(epoch, ema=True)
                if self.double_valid:
                    _ = self.validate(epoch, ema=False)
            else:
                eval_metrics = self.validate(epoch, ema=False)

            self.scheduler.step(epoch + 1, eval_metrics['Top1'])

            # save proper checkpoint with eval metric
            if self._master_node():
                torch.cuda.synchronize()
                best_metric, best_epoch = self.saver.save_checkpoint(epoch, metric=eval_metrics['Top1'])
                eval_metrics.update({'Best_Top1': best_metric})
                update_summary(epoch, train_metrics, eval_metrics, 'summary.csv', log_wandb=self.wandb)
        logging.info(f'*** Best Top1: {best_metric} {best_epoch} ***')

    def iterate(self, model, data, criterion):
        x, y = map(lambda x: x.to(self.device), data)
        x = x.to(memory_format=torch.channels_last)

        with autocast(enabled=True if self.scaler else False):
            prob = model(x)
            loss = criterion(prob, y)

        return loss, prob, y

    def train(self, epoch):
        self._reset_metric()
        num_updates = epoch * len(self.train_loader)
        pbar = self._set_pbar(self.train_loader)

        self.model.train()
        self.optimizer.zero_grad()
        for i, data in pbar:
            loss, prob, target = self.iterate(self.model, data, self.train_criterion)
            loss /= self.grad_accumulation

            self._backward(loss, (i + 1) % self.grad_accumulation == 0)

            self.losses.update(loss)
            computed_losses = self.losses.compute()
            if self._master_node():
                pbar.set_description(f'[Train#{epoch:>3}] Loss:{computed_losses:.6f}')

            torch.cuda.synchronize()
            num_updates += 1
            self.scheduler.step_update(num_updates=num_updates, metric=computed_losses)

        if hasattr(self.optimizer, 'sync_lookahead'):
            self.optimizer.sync_lookahead()

        return {'loss': self.losses.compute()}

    @torch.no_grad()
    def validate(self, epoch, ema=False):
        model = self.model_ema if ema else self.model
        log_prefix = ' EMA ' if ema else ' Val '
        self._reset_metric()
        pbar = self._set_pbar(self.val_loader)

        model.eval()
        for i, data in pbar:
            loss, prob, target = self.iterate(model, data, self.val_criterion)
            loss /= self.grad_accumulation
            self._update_metric(loss, prob, target)

            metrics = self._metrics()
            if self._master_node():
                pbar.set_description(self._print(metrics, epoch, log_prefix))

        return self._metrics()

    @torch.no_grad()
    def test(self, ema, test_loader):
        self._reset_metric()
        accuracies_list = list()
        pbar = self._set_pbar(test_loader)

        model = self.model_ema if ema else self.model

        model.eval()
        for i, data in pbar:
            loss, prob, target = self.iterate(model, data, self.val_criterion)
            self._update_metric(loss, prob, target)
            accuracies_list.append(accuracy(prob, target))

            metrics = self._metrics()
            if self._master_node():
                pbar.set_description(self._print(metrics, 0, 'Test'))

        return accuracies_list, self._metrics()

    def _backward(self, loss, update_grad):
        create_graph = hasattr(self.optimizer, 'is_second_order') and self.optimizer.is_second_order
        if self.scaler:
            self._scaler_backward(loss, create_graph, update_grad)
        else:
            self._default_backward(loss, create_graph, update_grad)

        if update_grad:
            self.optimizer.zero_grad()
            if self.model_ema:
                self.model_ema.update(self.model)

    def _scaler_backward(self, loss, create_graph, update_grad):
        self.scaler(loss, self.optimizer, self.clip_grad, self.clip_mode,
                    model_parameters(self.model, exclude_head='agc' in self.clip_mode), create_graph, update_grad)

    def _default_backward(self, loss, second_order, update_grad):
        loss.backward(create_graph=second_order)
        if update_grad:
            if self.clip_grad:
                dispatch_clip_grad(self.model.parameters(), self.clip_grad, mode=self.clip_mode)
            self.optimizer.step()

    def _set_pbar(self, loader):
        if self._master_node():
            pbar = tqdm(enumerate(loader), total=len(loader), dynamic_ncols=True)
        else:
            pbar = enumerate(loader)
        return pbar

    def _update_metric(self, loss, prob, target):
        self.losses.update(loss.item() / prob.size(0))
        self.top1.update(prob, target)
        self.top5.update(prob, target)

    def _reset_metric(self):
        self.losses.reset()
        self.top1.reset()
        self.top5.reset()

    def _metrics(self):
        return {
            'Loss': self.losses.compute(),
            'Top1': self.top1.compute() * 100,
            'Top5': self.top5.compute() * 100,
        }

    def _print(self, metrics, epoch, mode):
        log = f'[{mode}#{epoch:>3}] '
        for k, v in metrics.items():
            log += f'{k}:{v:.6f} | '
        return log[:-3]

    def _master_node(self):
        return self.local_rank == 0
