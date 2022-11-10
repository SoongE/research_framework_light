import hydra
import torch
import wandb
from omegaconf import DictConfig
from timm import create_model
from timm.utils import CheckpointSaver

from src.data.data_loader_v2 import load_dataloader_v2
from src.fit import Fit
from src.initial_setting import init_seed, init_distributed, init_logger, cuda_setting
from src.utils import model_tune, create_scheduler_v2, create_criterion, NativeScalerWithGradUpdate
from src.utils.benchmark import benchmark_model


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = cfg.set
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
    loaders = load_dataloader_v2(cfg)

    saver = CheckpointSaver(model=model, optimizer=optimizer, args=cfg, model_ema=model_ema, amp_scaler=scaler,
                            max_history=cfg.train.save_max_history)

    benchmark_result = benchmark_model(cfg.benchmark, model)
    wandb.log(benchmark_result)

    fit = Fit(cfg, scaler, device, start_epoch, num_epochs, model, criterions, optimizer, model_ema, scheduler, saver,
              loaders)

    fit()


if __name__ == "__main__":
    main()
