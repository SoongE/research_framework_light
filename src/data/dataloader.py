import warnings

from hydra import initialize, compose
from timm.data import create_dataset, create_loader, FastCollateMixup

warnings.filterwarnings("ignore")

mixup_args = {
    'mixup_alpha': 0.1,
    'cutmix_alpha': 1.0,
    'prob': 1.0,
    'switch_prob': 0.5,
    'mode': 'batch',
    'label_smoothing': 0,
    'num_classes': 1000}


def get_dataloader(cfg, *modes):
    res = []

    for mode in modes:
        is_training = True if mode == 'train' else False

        collate_fn = FastCollateMixup(**mixup_args) if cfg.train.mixup and is_training else None

        dataset = create_dataset(name=f'torch/{cfg.dataset.name}', root=cfg.dataset.root, split=mode,
                                 is_training=is_training, batch_size=cfg.train.batch_size)
        data_loader = create_loader(
            dataset,
            input_size=cfg.dataset.size,
            batch_size=cfg.train.batch_size,
            is_training=is_training,
            scale=cfg.dataset.scale,
            mean=cfg.dataset.mean,
            std=cfg.dataset.std,
            color_jitter=cfg.dataset.color_jitter,
            num_workers=cfg.train.num_workers,
            distributed=cfg.distributed,
            pin_memory=True,
            auto_augment=cfg.dataset.aa,
            # num_aug_repeats=cfg.dataset.aug_repeats,
            crop_pct=cfg.dataset.crop_pct,
            collate_fn=collate_fn,
            fp16=False,
        )

        res.append(data_loader)

    if len(modes) == 1:
        return res[0]
    else:
        return res


if __name__ == '__main__':
    with initialize(config_path='../../configs'):
        cfg = compose(config_name='config.yaml', overrides=['dataset=cifar100'])

    loader = get_dataloader(cfg, cfg.dataset.trainset_name)
    data = next(iter(loader))
    x, y = map(lambda x: x.to('cpu'), data)
    print(x.shape)
    print(y.shape)

    print(y)
