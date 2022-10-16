import os
import pickle
import warnings

import numpy
from hydra import initialize, compose
from torch.utils.data import Dataset, DistributedSampler
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets.samplers import DistributedSampler
from torchvision.transforms import transforms

from src.data import Collater

warnings.filterwarnings("ignore")


def get_cifar_dataloader(cfg, *modes):
    res = []

    for mode in modes:
        dataset = CustomDataset(mode, cfg.dataset.root, cfg.dataset.size)

        if cfg.distributed:
            sampler = DistributedSampler(dataset)
            shuffle = False
            drop_last = True if 'train' in mode else False
        else:
            sampler = None
            shuffle = True
            drop_last = False

        collate_fn = Collater(cfg.dataset.num_class) if cfg.train.mixup and 'train' in mode else None

        data_loader = DataLoader(dataset, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers,
                                 shuffle=shuffle, sampler=sampler, drop_last=drop_last, pin_memory=True,
                                 collate_fn=collate_fn)

        res.append(data_loader)

    if len(modes) == 1:
        return res[0]
    else:
        return res


class CustomDataset(Dataset):
    def __init__(self, mode, root, size):
        super().__init__()
        self.root = os.path.join(root, 'cifar-100-python')
        self.mode = mode
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                         std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])

        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(size, padding=4),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                normalize,
            ])

        with open(os.path.join(self.root, f'{mode}'), 'rb') as f:
            dataset = pickle.load(f, encoding='bytes')

        self.x = dataset['data'.encode()]
        self.y = dataset['fine_labels'.encode()]

    def __getitem__(self, index):
        r = self.x[index, :1024].reshape(32, 32)
        g = self.x[index, 1024:2048].reshape(32, 32)
        b = self.x[index, 2048:].reshape(32, 32)
        img = numpy.dstack((r, g, b))
        x = self.transform(img)

        return x, self.y[index]

    def __len__(self):
        return len(self.x)


if __name__ == '__main__':
    with initialize(config_path='../../configs'):
        cfg = compose(config_name='config.yaml', overrides=['dataset=cifar100'])

    loader = get_cifar_dataloader(cfg, cfg.dataset.trainset_name)
    data = next(iter(loader))
    x, y = map(lambda x: x.to('cpu'), data)
    print(x.shape)
    print(y.shape)
    print(y.type())
