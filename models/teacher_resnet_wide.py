"""
Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['WideResNet', 'wide_resnet26']

pretrained_path = {
    'wide_resnet26': 'checkpoints/wide_resnet26.pth.tar'
}

def conv3x3(in_planes, out_planes, stride=1, use_bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=use_bias)

class BasicWideResBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicWideResBlock, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Conv1
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride, use_bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Conv2
        self.conv2 = conv3x3(out_channels, out_channels, use_bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # ConvRes
        if self.stride > 1:
            self.avgpool = nn.AvgPool2d(2)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))

        res = x
        if self.stride > 1:
            res = self.avgpool(x)
        if self.in_channels != self.out_channels:
            res = torch.cat((res, res*0),1)

        return F.relu(y + res, inplace=True)

    def num_params(self):
        n_params = 0.
        for p in self.parameters():
            n_params += np.prod(p.size())
        return n_params

    def FLOPS(self, in_shape):
        spatial_dim = float(in_shape[1] * in_shape[2]) / (float(self.stride)**2.)
        flops =  spatial_dim * self.in_channels * self.out_channels * 9
        flops += spatial_dim * self.out_channels * 2
        flops += spatial_dim * self.out_channels * self.out_channels * 9
        flops += spatial_dim * self.out_channels * 2
        return flops


class WideResNet(nn.Module):
    def __init__(self, block, nblocks, num_classes=10):
        super(WideResNet, self).__init__()
        blocks = [block, block, block]
        factor = 1
        self.num_layers = nblocks
        self.in_planes = int(32*factor)
        self.conv1 = conv3x3(3, int(32*factor), stride=1)
        self.bn1 = nn.BatchNorm2d(int(32*factor))
        layer1 = self._make_layer(blocks[0], int(64*factor), nblocks[0], stride=2)
        layer2 = self._make_layer(blocks[1], int(128*factor), nblocks[1], stride=2)
        layer3 = self._make_layer(blocks[2], int(256*factor), nblocks[2], stride=2)
        self.layers = nn.ModuleList(layer1+layer2+layer3)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.ends_bn = nn.BatchNorm2d(int(256*factor))
        self.linear = nn.Linear(int(256*factor), num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, planes, nblocks, stride=1):
        layers = [block(self.in_planes, planes, stride)]
        self.in_planes = planes
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, planes))
        return layers

    def forward(self, x):
        ends = []
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        for l in self.layers:
            x = l(x)
            ends.append(x.detach())
        x = self.ends_bn(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x, ends

    def load_pretrained(self, fn):
        state = self.state_dict()
        checkpoint = torch.load(fn)['state_dict']
        checkpoint['linear.weight'] = state['linear.weight']
        checkpoint['linear.bias'] = state['linear.bias']
        self.load_state_dict(checkpoint)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False


def wide_resnet26(pretrained=True, **kwargs):
    """Constructs a WideResNet-26 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = WideResNet(BasicWideResBlock, [4, 4, 4], **kwargs)
    if pretrained:
        path = pretrained_path['wide_resnet26']
        model.load_pretrained(path)
    return model

def create_teacher(arch, pretrained=True, **kwargs):
    return eval(arch)(pretrained=pretrained, **kwargs)
