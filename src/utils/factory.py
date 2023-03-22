import timm
import torch
from lion_pytorch import Lion
from timm.optim import Lamb
from torch import nn
from torch.cuda.amp import GradScaler
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR, LambdaLR, MultiStepLR, \
    OneCycleLR, SequentialLR


class ObjectFactory:
    def __init__(self, cfg):
        self.cfg = cfg
        self.train = cfg.train
        self.optim = cfg.train.optimizer
        self.scheduler = cfg.train.lr_scheduler
        self.dataset = cfg.dataset
        self.model = cfg.model
        self.device = torch.device(f'cuda:{cfg.local_rank}') if torch.cuda.is_available() else torch.device('cpu')

    def create_model(self):
        model = timm.create_model(
            **self.model,
            in_chans=self.dataset.in_channels,
            num_classes=self.dataset.num_classes,
        )
        model.to(self.device)

        if self.train.channels_last:
            model = model.to(memory_format=torch.channels_last)

        if self.cfg.distributed and self.train.sync_bn:
            self.train.dist_bn = ''
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if self.model.scriptable:
            assert not self.train.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
            model = torch.jit.script(model)

        return torch.compile(model)

    def create_optimizer_and_scheduler(self, model, iter_per_epoch):
        self.cfg.train.iter_per_epoch = iter_per_epoch
        self.train.iter_per_epoch = iter_per_epoch

        optim = self.optim.optim
        sched = self.scheduler.sched

        parameter = model.parameters()
        total_iter = self.train.epochs * self.train.iter_per_epoch
        warmup_iter = self.scheduler.warmup_epochs * self.train.iter_per_epoch
        lr = self.optim.lr
        weight_decay = self.optim.weight_decay

        if optim == 'sgd':
            optimizer = SGD(parameter, lr, self.optim.momentum, weight_decay=weight_decay, nesterov=self.optim.nesterov)
        elif optim == 'adamw':
            optimizer = AdamW(parameter, lr, weight_decay=weight_decay, betas=self.optim.betas, eps=self.optim.eps)
        elif optim == 'lion':
            optimizer = Lion(parameter, lr, weight_decay=self.optim.weight_decay)
        elif optim == 'lamb':
            optimizer = Lamb(parameter, lr, weight_decay=weight_decay, betas=self.optim.betas, eps=self.optim.eps)
        else:
            NotImplementedError(f"{optim} is not supported yet")

        if sched == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, total_iter - warmup_iter, self.scheduler.min_lr)
        elif sched == 'multistep':
            scheduler = MultiStepLR(optimizer, [epoch * iter_per_epoch for epoch in self.scheduler.milestones])
        elif sched == 'step':
            scheduler = StepLR(optimizer, total_iter - warmup_iter, gamma=self.scheduler.decay_rate)
        elif sched == 'explr':
            scheduler = ExponentialLR(optimizer, gamma=self.scheduler.decay_rate)
        elif sched == 'onecyclelr':
            scheduler = OneCycleLR(optimizer, lr, total_iter)
        else:
            NotImplementedError(f"{sched} is not supported yet")

        if self.scheduler.warmup_epochs and sched != 'onecyclelr':
            if self.scheduler.warmup_scheduler == 'linear':
                lr_lambda = lambda e: (e * (
                        lr - self.scheduler.warmup_lr) / warmup_iter + self.scheduler.warmup_lr) / lr
                warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
            else:
                NotImplementedError(f"{self.scheduler.warmup_scheduler} is not supported yet")

            scheduler = SequentialLR(optimizer, [warmup_scheduler, scheduler], [warmup_iter])

        return optimizer, scheduler

    def create_criterion_scaler(self):
        if self.train.criterion in ['ce', 'crossentropy']:
            criterion = nn.CrossEntropyLoss()
        elif self.train.criterion in ['bce', 'binarycrossentropy']:
            criterion = BCEWithLogitsLossWithTypeCasting()
        elif self.train.criterion in ['ml']:
            criterion = nn.MultiLabelSoftMarginLoss()
        elif self.train.criterion in ['mse', 'l2']:
            criterion = MSELoss()
        else:
            NotImplementedError(f"{self.train.criterion} is not supported yet")

        if self.train.amp:
            scaler = NativeScalerWithGradAccum()
        else:
            scaler = None

        return criterion, scaler


class NativeScalerWithGradAccum:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = GradScaler()

    def __call__(self, loss, optimizer, model_param, clip_grad=None, update=True):
        self._scaler.scale(loss).backward()
        if update:
            if clip_grad:
                self._scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_param, clip_grad)
            self._scaler.step(optimizer)
            self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


class BCEWithLogitsLossWithTypeCasting(BCEWithLogitsLoss):
    def forward(self, y_hat, y):
        y = y.float()
        y = y.reshape(y_hat.shape)
        return super().forward(y_hat, y)
