import logging

import torch
import wandb
from easydict import EasyDict
from timm.loss import BinaryCrossEntropy, SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.models import convert_splitbn_model, resume_checkpoint as timm_resume_checkpoint, load_checkpoint
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ModelEmaV2, dispatch_clip_grad
from torch import nn
from torch.nn.parallel import DistributedDataParallel


def print_pass(*args):
    pass


def model_tune(model, scaler, device, cfg):
    # enable split bn (separate bn stats per batch-portion)
    if cfg.train.split_bn:
        assert cfg.train.argumentation.aug_splits > 1 or cfg.train.argumentation.resplit
        model = convert_splitbn_model(model, max(cfg.train.argumentation.aug_splits, 2))

    model.to(device)

    if cfg.train.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if cfg.distributed and cfg.train.sync_bn:
        assert not cfg.train.split_bn
        cfg.train.dist_bn = ''
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if cfg.model.scriptable:
        assert not cfg.train.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=EasyDict({**cfg.train.lr, **cfg.train.optimizer})))

    resume_epochs = resume_checkpoint(model, optimizer, scaler, cfg.train.resume, cfg.train.resume_opt,
                                      cfg.local_rank) if cfg.train.resume else None

    # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
    model_ema = ModelEmaV2(model, decay=cfg.train.model_ema_decay,
                           device='cpu' if cfg.model_ema_force_cpu else None) if cfg.train.model_ema else None

    load_checkpoint(model_ema.module, cfg.resume, use_ema=cfg.train.model_ema) if cfg.train.resume else None

    if cfg.distributed:
        model = DistributedDataParallel(model, device_ids=[cfg.local_rank], broadcast_buffers=cfg.train.ddp_bb)

    return model, optimizer, model_ema, resume_epochs


def resume_checkpoint(model, optimizer, loss_scaler, resume, resume_opt, local_rank):
    return timm_resume_checkpoint(
        model, resume,
        optimizer=optimizer if resume_opt else None,
        loss_scaler=loss_scaler if resume_opt else None,
        log_info=local_rank == 0
    )


def create_scheduler_v2(cfg, optimizer, resume_epochs):
    lr_params = dict(cfg.train.lr)
    lr_params.update({'epochs': cfg.train.epochs})
    lr_scheduler, num_epochs = create_scheduler(EasyDict(lr_params), optimizer)

    start_epochs = 0

    if resume_epochs:
        start_epochs = resume_epochs
        logging.info(f"RESUME epochs: {resume_epochs}")

    if lr_scheduler and start_epochs > 0:
        lr_scheduler.step(start_epochs)

    return lr_scheduler, num_epochs, start_epochs


def create_criterion(cfg, device):
    aug = cfg.dataset.augmentation
    mixup_active = aug.mixup > 0 or aug.cutmix > 0. or aug.cutmix_minmax is not None
    if mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if cfg.train.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=cfg.train.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif aug.smoothing:
        if cfg.train.bce_loss:
            train_loss_fn = BinaryCrossEntropy(smoothing=aug.smoothing,
                                               target_threshold=cfg.train.bce_target_thresh)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=aug.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.to(device)
    validate_loss_fn = nn.CrossEntropyLoss().to(device)
    return (train_loss_fn, validate_loss_fn)


class NativeScalerWithGradUpdate:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False,
                 update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
            self._scaler.step(optimizer)
            self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def reconstruct_cfg(cfg):
    if 'model' in cfg.train.keys():
        for k, v in cfg.train.model.items():
            cfg.model[k] = v
        cfg.train.model = {}


def logging_benchmark_result_to_wandb(benchmark_result, exp_name):
    columns = ['Name']
    columns.extend(list(benchmark_result.keys()))
    data = [exp_name]
    data.extend(list(benchmark_result.values()))
    wandb.log({'benchmark': wandb.Table(columns=columns, data=[data])})
