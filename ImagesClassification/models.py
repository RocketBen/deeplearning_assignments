import torch
import torchvision.models.resnet
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(BasicBlock, self).__init__()
        self.plain_block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, 1),  # 第一层stride=1则特征图大小不变，为2则下采样
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel)
        )
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, stride, 1),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.downsample = None
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x
        out = self.plain_block(x)
        out = self.relu(identity + out)
        return out


class BottleNeck(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(BottleNeck, self).__init__()
        self.plain_block = nn.Sequential(
            nn.Conv2d(in_channel, 64, 3, stride, 1),  # 第一层stride=1则特征图大小不变，为2则下采样
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel)
        )
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, stride, 1),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.downsample = None
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x
        out = self.plain_block(x)
        out = self.relu(identity + out)
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        # 224
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 1, 3),
            nn.BatchNorm2d(64),
            # 224
            nn.MaxPool2d(3, stride=2)
        )
        # 112
        self.conv2_x = BasicBlock(64, 64, 2)
        # 56
        self.conv3_x = BasicBlock(64, 128, 2)
        # 28
        self.conv4_x = BasicBlock(128, 256, 2)
        # 14
        self.conv5_x = BasicBlock(256, 512, 2)
        # 7
        self.avepool = nn.AvgPool2d(7)
        # 1
        self.fc = nn.Linear(512, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.avepool(out).squeeze()
        out = self.fc(out)
        return self.softmax(out)


class ResNet(nn.Module):
    def __init__(self, n):
        super(ResNet, self).__init__()
        # 32
        self.name = 'ResNet{}'.format(6 * n + 2)
        self.n = n
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
        )
        # 32
        self.conv2_x = self._make_layer(16, 32)
        # 16
        self.conv3_x = self._make_layer(32, 64)
        # 8
        self.conv4_x = self._make_layer(64, 128)
        # 4
        self.avepool = nn.AvgPool2d(4)
        # 1
        self.fc = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.avepool(out).squeeze()
        out = self.fc(out)
        return self.softmax(out)

    def _make_layer(self, in_channel, out_channel):
        # 产生n个块， 每个块有2层
        if self.n <= 5:  # 比较浅的网络用普通残差块
            layer_list = [BasicBlock(in_channel, in_channel, 1) for i in range(self.n - 1)]
            layer_list.append(BasicBlock(in_channel, out_channel, 2))
        else:  # 搭建深层网络用瓶颈层
            layer_list = [BottleNeck(in_channel, in_channel, 1) for i in range(self.n - 1)]
            layer_list.append(BottleNeck(in_channel, out_channel, 2))
        order_dict = OrderedDict([('block{}'.format(i), block) for i, block in enumerate(layer_list)])
        return nn.Sequential(order_dict)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight)


if __name__ == '__main__':
    net = ResNet18()
    imgs = torch.randn(256, 3, 224, 224)
    pred = net(imgs)
