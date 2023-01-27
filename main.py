import hydra
import torch
from omegaconf import DictConfig
from timm import create_model
from timm.utils import CheckpointSaver

from src.data import get_dataloader
from src.fit import Fit
from src.initial_setting import init_seed, init_distributed, init_logger, cuda_setting
from src.utils import model_tune, create_scheduler_v2, create_criterion, NativeScalerWithGradUpdate, \
    logging_benchmark_result_to_wandb, benchmark_model
from src.models import *


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cuda_setting(cfg.gpus)
    init_distributed(cfg)
    init_seed(cfg.train.seed + cfg.local_rank)

    device = torch.device(f'cuda:{cfg.local_rank}') if torch.cuda.is_available() else torch.device('cpu')

    init_logger(cfg)

    model = create_model(**cfg.model, num_classes=cfg.dataset.num_classes)
    criterions = create_criterion(cfg, device)
    scaler = NativeScalerWithGradUpdate()

    model, optimizer, model_ema, resume_epochs = model_tune(model, scaler, device, cfg)

    scheduler, num_epochs, start_epoch = create_scheduler_v2(cfg, optimizer, resume_epochs)
    loaders = get_dataloader(cfg)

    saver = CheckpointSaver(model=model, optimizer=optimizer, args=cfg, model_ema=model_ema, amp_scaler=scaler,
                            max_history=cfg.train.save_max_history)

    if cfg.local_rank == 0 and cfg.wandb:
        benchmark_result = benchmark_model(cfg.benchmark, model)
        logging_benchmark_result_to_wandb(benchmark_result, cfg.name)

    fit = Fit(cfg, scaler, device, start_epoch, num_epochs, model, criterions, optimizer, model_ema, scheduler, saver,
              loaders)

    fit()


if __name__ == "__main__":
    main()
