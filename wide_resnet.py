import os
import torch
import torch.nn as nn
import torch.nn.functional as F

PRETRAINED_PATH = {
    'wide_resnet26': 'checkpoints/wide_resnet26.pth.tar'
}

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)  

# No projection: identity shortcut
class BasicWideBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, shortcut=False):
        super(BasicWideBlock, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        
        if self.stride > 1:
            self.avgpool = nn.AvgPool2d(2)
        
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)

        y = self.conv2(y)
        y = self.bn2(y)

        res = x
        if self.stride > 1:
            res = self.avgpool(x)
        if self.in_planes != self.planes:
            res = torch.cat((res, res*0),1)
        return F.relu(y + res, inplace=True)


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
        shortcut = 0
        if stride != 1 or self.in_planes != planes:
            shortcut = 1
        layers = [block(self.in_planes, planes, stride, shortcut)]
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
        checkpoint['classifier.weight'] = state['classifier.weight']
        checkpoint['classifier.bias'] = state['classifier.bias']
        loaded_tensors = []

        # self.load_state_dict(state)
        return loaded_tensors

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False


def wide_resnet26(num_classes=10, pretrained=True):
    model = WideResNet(BasicWideBlock, [4,4,4], num_classes)
    if pretrained:
        path = PRETRAINED_PATH['wide_resnet26']
        if not os.path.isfile(path):
            raise ValueError('\nMissing model weights. Please run:\nmkdir -p checkpoints && cd checkpoints && python download_models.py && cd ..')
        model.load_pretrained(path)
    return model


if __name__=='__main__':
    import proj_utils
    model = wide_resnet26(10, pretrained=True)
    print(model)
    print(proj_utils.parameter_description(model))
    x = torch.zeros((5,3,64,64))
    y = model(x)