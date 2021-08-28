import os
import numpy as np
import pickle
import torch
from torch.utils import data
from torch import nn
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm
from torch.nn import functional as F


class Cifar10Dataset(data.Dataset):
    def __init__(self, is_train=True, transform=None):
        root = r'cifar-10'
        self.transform = transform
        self.labels, self.imgs = [], []
        if is_train:
            for i in range(1, 6):
                with open(os.path.join(root, 'data_batch_{}'.format(i)), 'rb') as f:
                    data_dict = pickle.load(f, encoding='bytes')
                    self.imgs.append(data_dict[b'data'])
                    self.labels.append(data_dict[b'labels'])
        else:
            with open(os.path.join(root, 'test_batch'), 'rb') as f:
                data_dict = pickle.load(f, encoding='bytes')
                self.imgs.append(data_dict[b'data'])
                self.labels.append(data_dict[b'labels'])
        self.imgs = np.concatenate(self.imgs, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

    def __getitem__(self, index):
        img, label = self.imgs[index, :], self.labels[index]
        img = img.reshape((3, 32, 32))
        img = np.transpose(img, (2, 1, 0))
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.labels)


def model_test(model, device, test_loader, test_len=None):
    cnt = 0
    total = 0
    ave_loss = 0
    if not test_len:
        test_len = len(test_loader)
    else:
        test_len = min(test_len, len(test_loader))
    with torch.no_grad():
        for i, (imgs, labels) in tqdm(enumerate(test_loader), total=test_len, desc='test', leave=False):
            if i == test_len:
                break
            imgs, labels = imgs.to(device), labels.to(device)
            prob = model(imgs)
            loss = F.cross_entropy(prob, labels)
            pred = torch.argmax(prob, dim=1)
            cnt += torch.sum((pred == labels), dim=0).item()
            total += pred.shape[0]
            ave_loss += loss.item()
    return cnt / total, ave_loss / test_len


if __name__ == '__main__':
    from time import time

    t0 = time()
    train_set = Cifar10Dataset(is_train=False)
    print(time() - t0)
    x, y = train_set[0]
    plt.figure()
    plt.imshow(x)
    plt.show()
