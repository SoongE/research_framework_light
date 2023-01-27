from timm.data import create_dataset, FastCollateMixup, Mixup, AugMixDataset, create_loader, resolve_data_config


def load_dataloader_v2(cfg):
    aug = cfg.train.augmentation
    dataset = cfg.dataset

    data_config = resolve_data_config(aug)

    dataset_train = create_dataset(
        dataset.name, root=dataset.root, split=dataset.train_name, is_training=True,
        class_map=dataset.class_map,
        batch_size=cfg.train.batch_size,
        repeats=aug.epoch_repeats)
    dataset_eval = create_dataset(
        dataset.name, root=dataset.root, split=dataset.valid_name, is_training=False,
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

    # create data loaders w/ augmentation pipeiine
    train_interpolation = aug.train_interpolation

    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=cfg.train.batch_size,
        is_training=True,
        use_prefetcher=aug.prefetcher,
        no_aug=aug.no_aug,
        re_prob=aug.reprob,
        re_mode=aug.remode,
        re_count=aug.recount,
        re_split=aug.resplit,
        scale=dataset.scale,
        ratio=aug.ratio,
        hflip=aug.hflip,
        vflip=aug.vflip,
        color_jitter=aug.color_jitter,
        auto_augment=aug.aa,
        num_aug_repeats=aug.aug_repeats,
        num_aug_splits=aug.aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=cfg.train.num_workers,
        distributed=cfg.distributed,
        collate_fn=collate_fn,
        pin_memory=aug.pin_mem,
        use_multi_epochs_loader=aug.use_multi_epochs_loader,
        worker_seeding=aug.worker_seeding,
    )

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=cfg.train.batch_size,
        is_training=False,
        use_prefetcher=aug.prefetcher,
        interpolation=aug.test_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=cfg.train.num_workers,
        distributed=cfg.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=aug.pin_mem,
    )
    return (loader_train, loader_eval)
