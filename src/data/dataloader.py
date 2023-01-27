from timm.data import create_dataset, FastCollateMixup, Mixup, AugMixDataset, create_loader

from src.data import get_cifar_dataloader


def base_dataloader(cfg):
    aug = cfg.dataset.augmentation
    dataset = cfg.dataset

    dataset_train = create_dataset(
        dataset.name, root=dataset.root, split=dataset.train, is_training=True,
        class_map=dataset.class_map,
        batch_size=cfg.train.batch_size,
        repeats=aug.epoch_repeats)
    dataset_eval = create_dataset(
        dataset.name, root=dataset.root, split=dataset.valid, is_training=False,
        class_map=dataset.class_map,
        batch_size=cfg.train.batch_size)

    collate_fn = None
    mixup_active = aug.mixup > 0 or aug.cutmix > 0. or aug.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=aug.mixup, cutmix_alpha=aug.cutmix, cutmix_minmax=aug.cutmix_minmax,
            prob=aug.mixup_prob, switch_prob=aug.mixup_switch_prob, mode=aug.mixup_mode,
            label_smoothing=aug.smoothing, num_classes=dataset.num_classes)
        if aug.prefetcher:
            assert not aug.aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            collate_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if aug.aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=cfg.aug_splits)
    loader_train = create_loader(
        dataset_train,
        input_size=tuple(dataset.size),
        batch_size=cfg.train.batch_size,
        is_training=True,
        use_prefetcher=aug.prefetcher,
        no_aug=aug.no_aug,
        re_prob=aug.reprob,
        re_mode=aug.remode,
        re_count=aug.recount,
        re_split=aug.resplit,
        scale=aug.scale,
        ratio=aug.ratio,
        hflip=aug.hflip,
        vflip=aug.vflip,
        color_jitter=aug.color_jitter,
        auto_augment=aug.aa,
        num_aug_repeats=aug.aug_repeats,
        num_aug_splits=aug.aug_splits,
        interpolation=aug.train_interpolation,
        mean=tuple(aug.mean),
        std=tuple(aug.std),
        num_workers=cfg.train.num_workers,
        distributed=cfg.distributed,
        collate_fn=collate_fn,
        pin_memory=aug.pin_mem,
        use_multi_epochs_loader=aug.use_multi_epochs_loader,
        worker_seeding=aug.worker_seeding,
    )

    loader_eval = create_loader(
        dataset_eval,
        input_size=tuple(dataset.size),
        batch_size=cfg.train.batch_size,
        is_training=False,
        use_prefetcher=aug.prefetcher,
        interpolation=aug.test_interpolation,
        mean=tuple(aug.mean),
        std=tuple(aug.std),
        num_workers=cfg.train.num_workers,
        distributed=cfg.distributed,
        crop_pct=aug.crop_pct,
        pin_memory=aug.pin_mem,
    )
    return (loader_train, loader_eval)


def get_dataloader(cfg):
    if 'cifar' in cfg.dataset.name:
        return get_cifar_dataloader(cfg)
    return base_dataloader(cfg)
