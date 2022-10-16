import builtins
import os
import random

import numpy as np
import torch
from torch.distributed import init_process_group

from src.utils import Logger, print_pass


def init_logger(cfg):
    if cfg.local_rank == 0:
        logger = Logger(cfg, cfg.wandb)
        return logger


def init_distributed(distributed, ngpus_per_node, local_rank, port, cfg):
    cfg.local_rank = local_rank
    if not distributed:
        return

    init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{port}',
        world_size=ngpus_per_node,
        rank=local_rank)
    if local_rank != 0:
        builtins.print = print_pass
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    cfg.world_size = torch.distributed.get_world_size()


def cuda_setting(gpus):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in gpus)


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
