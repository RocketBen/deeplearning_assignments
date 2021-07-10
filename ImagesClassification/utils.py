import os
import numpy as np
import pickle
import torch
from torch.utils import data
from torch import nn


def load_CIFAR10(root=r'cifar-10'):
    train_x, train_y = None, []
    for i in range(1, 6):
        with open(os.path.join(root, 'data_batch_{}'.format(i)), 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
            train_x = np.append(train_x, data_dict[b'data'], axis=0) if train_x is not None else data_dict[b'data']
            train_y += data_dict[b'labels']

    test_x, test_y = [], []
    with open(os.path.join(root, 'test_batch'), 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
        test_x = data_dict[b'data']
        test_y = data_dict[b'labels']

    return (train_x, train_y), (test_x, test_y)


class Cifar10Dataset(data.Dataset):
    def __init__(self, is_train=True, ont_hot=False, root=r'cifar-10', transform=None):
        self.one_hot = ont_hot
        self.num_classes = 10
        self.transform = transform
        if is_train:
            train_x, train_y = None, []
            for i in range(1, 6):
                with open(os.path.join(root, 'data_batch_{}'.format(i)), 'rb') as f:
                    data_dict = pickle.load(f, encoding='bytes')
                    train_x = np.append(train_x, data_dict[b'data'], axis=0) if train_x is not None else data_dict[b'data']
                    train_y += data_dict[b'labels']
            self.imgs, self.labels = train_x, train_y
        else:
            test_x, test_y = [], []
            with open(os.path.join(root, 'test_batch'), 'rb') as f:
                data_dict = pickle.load(f, encoding='bytes')
                test_x = data_dict[b'data']
                test_y = data_dict[b'labels']
            self.imgs, self.labels = test_x, test_y

    def __getitem__(self, index):
        img, label = self.imgs[index], self.labels[index]
        # reshape as 3x32x32
        img = img.reshape((32, 32, 3))
        if self.transform:
            img = self.transform(img)
        if self.one_hot:
            # change label to ont-hot format if needed
            label = torch.zeros(self.num_classes).scatter_(0, torch.LongTensor([label, ]), 1).to(torch.long)
        return img, label

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    train_set = Cifar10Dataset(is_train=False)
