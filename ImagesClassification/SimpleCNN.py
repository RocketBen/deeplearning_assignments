import numpy as np
import torchvision.models

from utils import Cifar10Dataset
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.utils import data
from torchvision import transforms as T
from tqdm import tqdm


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=0):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),

        )

    def forward(self, x):
        return self.layer(x)


class Net(nn.Module):
    def __init__(self, img_size=32):
        super(Net, self).__init__()
        # img_sizes = [32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2]
        channels = [3, 6, 9, 12, 16, 20, 24, 26, 30, 36, 40, 48, 56, 64, 80, 96]
        self.convs = []
        for i in range(len(channels))[:-1]:
            self.convs.append(ConvBlock(channels[i], channels[i + 1]))
        self.convs.append(
            nn.Sequential(
                nn.Conv2d(96, 182, 2, 1, 0),
                nn.BatchNorm2d(182),
                nn.ReLU()
            )
        )
        self.convs = nn.ModuleList(self.convs)
        self.linears = [
            nn.Linear(182, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 10),
            nn.Softmax(dim=1)]
        self.linears = nn.ModuleList(self.linears)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)

        x = x.view(x.shape[0], -1)

        for linear in self.linears:
            x = linear(x)
        return x


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
network = Net().to(device)
lr = 1e-3
LossFunc = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(network.parameters(), lr)
epochs = 100
batch_size = 64

train_transform = T.Compose([T.ToTensor(), 
          T.RandomHorizontalFlip(),
          T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

test_transform = T.Compose([T.ToTensor(), 
          T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

train_loader = data.DataLoader(dataset=Cifar10Dataset(is_train=True, transform=train_transform),
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=2)
test_loader = data.DataLoader(dataset=Cifar10Dataset(is_train=False, transform=test_transform),
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2)

if __name__ == '__main__':
    # train
    print(network)
    best_accuracy = 0
    for epoch in range(epochs):
        for idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = network(imgs)
            loss = LossFunc(preds, labels)
            loss.backward()
            optimizer.step()
            if idx % 100 == 0:
                print('{:.4f} %    loss:{:.4f}'.format(
                    100 * (epoch * len(train_loader) + idx) / (len(train_loader) * epochs),
                    loss.data))
        # test for each epoch
        print('testing for epoch {}'.format(epoch))
        cnt = 0
        test_loss = 0
        with torch.no_grad():
            for idx, (imgs, labels) in enumerate(test_loader):
                imgs, labels = imgs.cuda(), labels.cuda()
                preds = network(imgs)
                test_loss += LossFunc(preds, labels).detach()
                preds = torch.argmax(preds, dim=1)
                cnt += torch.sum(preds == labels)
            accu = cnt / (len(test_loader) * batch_size)
            if accu > best_accuracy:
                best_accuracy = accu
                torch.save(network.state_dict(), 'best.pth')
            print('accuracy: {:.4f}%   aveloss: {:.4f}'.format(100 * accu,
                test_loss / len(test_loader)))
