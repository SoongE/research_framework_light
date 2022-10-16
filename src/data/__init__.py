from timm.data import Mixup
from torch.utils.data.dataloader import default_collate


class Collater:
    def __init__(self, num_classes):
        mixup_args = {
            'mixup_alpha': 0.1,
            'cutmix_alpha': 1.0,
            'prob': 1.0,
            'switch_prob': 0.5,
            'mode': 'batch',
            'label_smoothing': 0,
            'num_classes': num_classes}
        self.mix_fn = Mixup(**mixup_args)

    def __call__(self, batch):
        return self.mix_fn(*default_collate(batch))
