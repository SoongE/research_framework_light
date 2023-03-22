import wandb
from timm.models import resume_checkpoint as timm_resume_checkpoint, load_checkpoint
from timm.utils import ModelEmaV2
from torch.nn.parallel import DistributedDataParallel


def print_pass(*args, **kwargs):
    pass


def model_tune(model, cfg):
    model_ema = ModelEmaV2(model, decay=cfg.train.model_ema_decay,
                           device='cpu' if cfg.model_ema_force_cpu else None) if cfg.train.model_ema else None

    load_checkpoint(model_ema.module, cfg.resume, use_ema=cfg.train.model_ema) if cfg.train.resume else None

    if cfg.distributed:
        model = DistributedDataParallel(model, static_graph=False, device_ids=[cfg.local_rank], broadcast_buffers=cfg.train.ddp_bb)

    return model, model_ema


def resume_checkpoint(model, optimizer, loss_scaler, resume, resume_opt, local_rank):
    return timm_resume_checkpoint(
        model, resume,
        optimizer=optimizer if resume_opt else None,
        loss_scaler=loss_scaler if resume_opt else None,
        log_info=local_rank == 0
    )


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
