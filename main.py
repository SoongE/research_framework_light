import gc

import hydra
import torch
import wandb
from omegaconf import DictConfig
from timm.utils import CheckpointSaver

from src.data import get_dataloader
from src.engine import Engine
from src.initial_setting import init_seed, init_distributed, init_logger, cuda_setting
from src.utils import model_tune, ObjectFactory, logging_benchmark_result_to_wandb
from src.models import *

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cuda_setting(cfg.gpus)
    init_distributed(cfg)
    init_seed(cfg.train.seed + cfg.local_rank)

    device = torch.device(f'cuda:{cfg.local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    cfg.name = cfg.model.model_name if cfg.name == '' else cfg.name

    loaders = get_dataloader(cfg)

    factory = ObjectFactory(cfg)
    model = factory.create_model()
    optimizer, scheduler, n_epochs = factory.create_optimizer_and_scheduler(model, len(loaders[0]))
    criterion, scaler = factory.create_criterion_scaler()

    model, model_ema, start_epoch, scheduler = model_tune(model, optimizer, scaler, scheduler, cfg)

    cfg = factory.cfg
    init_logger(cfg)

    if cfg.do_benchmark and cfg.is_master and cfg.wandb and not cfg.train.resume:
        from src.utils.benchmark import benchmark_with_model
        benchmark_result = benchmark_with_model(cfg, model)
        logging_benchmark_result_to_wandb(benchmark_result, cfg.name)

    saver = CheckpointSaver(model=model, optimizer=optimizer, args=cfg, model_ema=model_ema, amp_scaler=scaler,
                            max_history=cfg.train.save_max_history)
    epochs = (start_epoch, n_epochs)

    engine = Engine(cfg, scaler, device, epochs, model, criterion, optimizer, model_ema, scheduler, saver, loaders)

    engine()

    if cfg.is_master:
        torch.cuda.empty_cache()
        gc.collect()
        wandb.finish(quiet=True)


if __name__ == "__main__":
    main()
