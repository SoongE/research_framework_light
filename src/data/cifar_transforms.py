""" Loader Factory, Fast Collate, CUDA Prefetcher

Prefetcher and Fast Collate inspired by NVIDIA APEX example at
https://github.com/NVIDIA/apex/commit/d5e2bb4bdeedd27b1dfaf5bb2b24d6c000dee9be#diff-cf86c282ff7fba81fad27a559379d5bf

Hacked together by / Copyright 2019, Ross Wightman
Add torch.Generator for Reproducibility in torch2.0
"""
import math

import torch.utils.data
from timm.data import str_to_pil_interp, rand_augment_transform, auto_augment_transform, ToNumpy, \
    str_to_interp_mode
from timm.data.auto_augment import augment_and_mix_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.random_erasing import RandomErasing
from timm.data.transforms_factory import transforms_noaug_train
from torchvision import transforms

CIFAR_DEFAULT_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR_DEFAULT_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


def transforms_cifar_train(
        img_size=32,
        hflip=0.5,
        vflip=0.0,
        color_jitter=0.4,
        auto_augment=None,
        interpolation='random',
        use_prefetcher=False,
        mean=CIFAR_DEFAULT_MEAN,
        std=CIFAR_DEFAULT_STD,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        separate=False,
        force_color_jitter=False,
):
    """
    If separate==True, the transforms are returned as a tuple of 3 separate transforms
    for use in a mixing dataset that passes
     * all data through the first (primary) transform, called the 'clean' data
     * a portion of the data through the secondary transform
     * normalizes and converts the branches above with the third, final transform
    """
    primary_tfl = [transforms.RandomCrop(img_size, padding=16)]
    if hflip > 0.:
        primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.:
        primary_tfl += [transforms.RandomVerticalFlip(p=vflip)]

    secondary_tfl = []
    disable_color_jitter = False
    if auto_augment:
        assert isinstance(auto_augment, str)
        # color jitter is typically disabled if AA/RA on,
        # this allows override without breaking old hparm cfgs
        disable_color_jitter = not (force_color_jitter or '3a' in auto_augment)
        if isinstance(img_size, (tuple, list)):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != 'random':
            aa_params['interpolation'] = str_to_pil_interp(interpolation)
        if auto_augment.startswith('rand'):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
        elif auto_augment.startswith('augmix'):
            aa_params['translate_pct'] = 0.3
            secondary_tfl += [augment_and_mix_transform(auto_augment, aa_params)]
        elif auto_augment.startswith('torchvision'):
            secondary_tfl += [transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10,
                                                     str_to_interp_mode(interpolation))]
        else:
            secondary_tfl += [auto_augment_transform(auto_augment, aa_params)]

    if color_jitter is not None and not disable_color_jitter:
        # color jitter is enabled when not using AA or when forced
        if isinstance(color_jitter, (list, tuple)):
            # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
            # or 4 if also augmenting hue
            assert len(color_jitter) in (3, 4)
        else:
            # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
            color_jitter = (float(color_jitter),) * 3
        secondary_tfl += [transforms.ColorJitter(*color_jitter)]

    final_tfl = []
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        final_tfl += [ToNumpy()]
    else:
        final_tfl += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ]
        if re_prob > 0.:
            final_tfl.append(
                RandomErasing(re_prob, mode=re_mode, max_count=re_count, num_splits=re_num_splits, device='cpu'))

    if separate:
        return transforms.Compose(primary_tfl), transforms.Compose(secondary_tfl), transforms.Compose(final_tfl)
    else:
        return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)


def transforms_cifar_eval(
        img_size=32,
        crop_pct=1.0,
        crop_mode=None,
        interpolation='bilinear',
        use_prefetcher=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD
):
    if isinstance(img_size, (tuple, list)):
        assert len(img_size) == 2
        scale_size = tuple([math.floor(x / crop_pct) for x in img_size])
    else:
        scale_size = math.floor(img_size / crop_pct)
        scale_size = (scale_size, scale_size)

    if crop_mode == 'squash':
        # squash mode scales each edge to 1/pct of target, then crops
        # aspect ratio is not preserved, no img lost if crop_pct == 1.0
        tfl = [
            transforms.Resize(scale_size, interpolation=str_to_interp_mode(interpolation)),
            transforms.CenterCrop(img_size),
        ]
    else:
        # default crop model is center
        # aspect ratio is preserved, crops center within image, no borders are added, image is lost
        if scale_size != img_size:
            # simple case, use torchvision built-in Resize w/ shortest edge mode (scalar size arg)
            tfl = [
                transforms.Resize(scale_size[0], interpolation=str_to_interp_mode(interpolation)),
                transforms.CenterCrop(img_size)
            ]
        else:
            tfl = [transforms.CenterCrop(img_size)]

    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        tfl += [ToNumpy()]
    else:
        tfl += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std),
            )
        ]

    return transforms.Compose(tfl)


def create_transform_cifar(
        input_size,
        is_training=False,
        use_prefetcher=False,
        no_aug=False,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        crop_pct=None,
        crop_mode=None,
        tf_preprocessing=False,
        scale=None,
        ratio=None,
        separate=False):
    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    if is_training and no_aug:
        assert not separate, "Cannot perform split augmentation with no_aug"
        transform = transforms_noaug_train(
            img_size,
            interpolation=interpolation,
            use_prefetcher=use_prefetcher,
            mean=mean,
            std=std,
        )
    elif is_training:
        transform = transforms_cifar_train(
            img_size,
            hflip=hflip,
            vflip=vflip,
            color_jitter=color_jitter,
            auto_augment=auto_augment,
            interpolation=interpolation,
            use_prefetcher=use_prefetcher,
            mean=mean,
            std=std,
            re_prob=re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits,
            separate=separate,
        )
    else:
        assert not separate, "Separate transforms not supported for validation preprocessing"
        transform = transforms_cifar_eval(
            img_size,
            interpolation=interpolation,
            use_prefetcher=use_prefetcher,
            mean=mean,
            std=std,
            crop_pct=crop_pct,
            crop_mode=crop_mode,
        )

    return transform
