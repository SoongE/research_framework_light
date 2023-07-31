from timm import create_model
from timm.models import register_model
from torch import nn


@register_model
def resnet50_cifar(pretrained=False, **kwargs):
    model = create_model('resnet50', pretrained=pretrained, **kwargs)
    conv1 = model.conv1

    model.conv1 = nn.Conv2d(conv1.in_channels, conv1.out_channels, 3, 1, 1, conv1.dilation, conv1.groups, conv1.bias)
    model.maxpool = nn.Identity()
    model.init_weights()

    return model

