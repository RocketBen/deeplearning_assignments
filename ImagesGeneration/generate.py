import torch
from torch.utils import data
from torch import nn
from utils import *
from models import *
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import torchvision

device = torch.device('cuda:2')
ckpt_path = '/home/clb/PycharmProjects/deeplearning_assiments/Images_Generation/DCGAN_11.pth'
net_G = Generator()


def draw_gif():
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path)
    save_imgs = ckpt['save_imgs'].cpu()
    fig = plt.figure()
    plt.axis('off')
    imgs = []
    for i in range(1, save_imgs.shape[0]):
        img = save_imgs[i]
        img = img.masked_fill(img < 0, 0)
        img = plt.imshow(img, animated=True)
        imgs.append([img])
    ani = animation.ArtistAnimation(fig, imgs, interval=400, repeat_delay=5000, blit=True)
    plt.show()
    ani.save('face.gif', writer='imagemagick')


def draw_grid():
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path)
    net_G.load_state_dict(ckpt['net_G'])
    net_G.to(device)
    plt.figure()
    noise = torch.randn(16, nz, 1, 1, device=device)
    imgs = net_G(noise).detach()
    imgs = torchvision.utils.make_grid(imgs, nrow=4, normalize=True)
    imgs_fake = imgs.permute((1, 2, 0)).cpu().numpy()
    train_loader = data.DataLoader(
        CelebaDataset(transforms=T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])), batch_size=16, pin_memory=True, shuffle=True, num_workers=4, drop_last=True
    )
    imgs = next(iter(train_loader))
    imgs = torchvision.utils.make_grid(imgs, nrow=4, normalize=True)
    imgs_real = imgs.permute((1, 2, 0)).cpu().numpy()
    plt.figure()
    plt.subplot(121)
    plt.title('real')
    plt.axis('off')
    plt.imshow(imgs_real)
    plt.subplot(122)
    plt.title('fake')
    plt.axis('off')
    plt.imshow(imgs_fake)
    plt.show()


if __name__ == '__main__':
    draw_grid()
    draw_gif()
