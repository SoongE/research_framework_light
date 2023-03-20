import hydra
import torch
from omegaconf import DictConfig
from timm.utils import CheckpointSaver

from src.data import get_dataloader
from src.fit import Fit
from src.initial_setting import init_seed, init_distributed, init_logger, cuda_setting
from src.utils import model_tune, logging_benchmark_result_to_wandb, benchmark_model
from src.utils.factory import ObjectFactory


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cuda_setting(cfg.gpus)
    init_distributed(cfg)
    init_seed(cfg.train.seed + cfg.local_rank)

    device = torch.device(f'cuda:{cfg.local_rank}') if torch.cuda.is_available() else torch.device('cpu')

    init_logger(cfg)

    loaders = get_dataloader(cfg)
    factory = ObjectFactory(cfg)

    model = factory.create_model()
    optimizer, scheduler = factory.create_optimizer_and_scheduler(model, len(loaders[0]))
    criterion, scaler = factory.create_criterion_scaler()

    model, model_ema = model_tune(model, cfg)

    saver = CheckpointSaver(model=model, optimizer=optimizer, args=cfg, model_ema=model_ema, amp_scaler=scaler,
                            max_history=cfg.train.save_max_history)

    if cfg.local_rank == 0 and cfg.wandb:
        benchmark_result = benchmark_model(cfg.benchmark, model)
        logging_benchmark_result_to_wandb(benchmark_result, cfg.name)

    cfg = factory.cfg
    fit = Fit(cfg, scaler, device, cfg.train.epochs, model, criterion, optimizer, model_ema, scheduler, saver, loaders)

    fit()


if __name__ == "__main__":
    main()
