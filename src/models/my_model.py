from timm.models import register_model
from torch import nn


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1000),
        )

    def forward(self, x):
        return self.seq(x)


@register_model
def simple_net(pretrained=False, **kwargs):
    return SimpleNet()
