import hydra
import torch
from omegaconf import DictConfig
from timm import create_model
from timm.utils import CheckpointSaver, setup_default_logging

from src.data.data_loader_v2 import load_dataloader_v2
from src.fit import Fit
from src.initial_setting import init_seed, init_distributed, init_logger, cuda_setting
from src.utils import model_tune, create_scheduler_v2, create_criterion, NativeScalerWithGradUpdate


def main_worker(local_rank, ngpus_per_node, cfg):
    # print(OmegaConf.to_yaml(cfg))

    device = torch.device(f'cuda:{local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    init_seed(cfg.train.seed + local_rank)
    init_distributed(cfg.distributed, ngpus_per_node, local_rank, cfg.port, cfg)
    init_logger(cfg)

    model = create_model(**cfg.model, num_classes=cfg.dataset.num_classes)
    criterions = create_criterion(cfg, device)
    scaler = NativeScalerWithGradUpdate()

    model, optimizer, model_ema, resume_epochs = model_tune(model, scaler, local_rank, device, cfg)

    scheduler, num_epochs, start_epoch = create_scheduler_v2(cfg, optimizer, resume_epochs)
    loaders = load_dataloader_v2(cfg)

    saver = CheckpointSaver(model=model, optimizer=optimizer, args=cfg, model_ema=model_ema, amp_scaler=scaler,
                            max_history=cfg.train.save_max_history)

    fit = Fit(cfg, scaler, device, local_rank, start_epoch, num_epochs, model, criterions, optimizer, model_ema,
              scheduler, saver, loaders)

    fit()


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = cfg.set
    cuda_setting(cfg.gpus)
    ngpus_per_node = len(cfg.gpus)

    cfg.distributed = ngpus_per_node > 1

    if cfg.distributed:
        torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg))
    else:
        main_worker(cfg.gpus[0], ngpus_per_node, cfg)


if __name__ == "__main__":
    main()
