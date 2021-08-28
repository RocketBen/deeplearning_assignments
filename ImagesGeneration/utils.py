from torch.utils import data
from os.path import join
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms as T

root = './CelebA'


class CelebaDataset(data.Dataset):
    def __init__(self, transforms=None):
        super(CelebaDataset, self).__init__()
        self.transformes = transforms
        self.img_list = []
        img_path = join(root, 'Img', 'img_align_celeba')
        for _, _, files in os.walk(img_path):
            for img_name in files:
                self.img_list.append(join(img_path, img_name))

    def __getitem__(self, index):
        img = Image.open(self.img_list[index])
        if self.transformes is not None:
            img = self.transformes(img)
        return img

    def __len__(self):
        return len(self.img_list)


if __name__ == '__main__':
    trainset = CelebaDataset(transforms=T.Compose([
        T.Resize(64),
        T.CenterCrop(64),
        T.ToTensor(),
        #T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))
    trainloader = data.DataLoader(trainset, batch_size=4)
    x = next(iter(trainloader))
    x = x.numpy().transpose((0, 2, 3, 1)) * 0.5 + 0.5
    plt.figure()
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.axis('off')
        plt.imshow(x[i])
    plt.show()
