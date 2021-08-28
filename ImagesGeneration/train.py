import torch
from torch.utils import data
from torch import nn
from utils import *
from models import *
from tqdm import tqdm
from matplotlib import pyplot as plt
import threading

epochs = 30
lr_D = 0.0002
lr_G = 0.00005
beta1 = 0.5
batch_size = 256
device = torch.device('cuda:2')
ckpt_path = '/home/clb/PycharmProjects/deeplearning_assiments/Images_Generation/DCGAN_17.pth'

visulize = True
save = True

train_loader = data.DataLoader(
    CelebaDataset(transforms=T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])), batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=4, drop_last=True
)

net_G = Generator()
net_G.apply(weights_init)
net_D = Discriminator()
net_D.apply(weights_init)
net_G.to(device)
net_D.to(device)

criterion = nn.BCELoss()
optimizerD = torch.optim.Adam(net_D.parameters(), lr=lr_D, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(net_G.parameters(), lr=lr_G, betas=(beta1, 0.999))

save_imgs = torch.randn(1, image_size, image_size, 3)


def visulization():
    img = fake[0].detach()
    img = img * 0.5 + 0.5
    img = img.masked_fill(img < 0, 0).permute((1, 2, 0)).cpu()
    plt.figure()
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    global save_imgs
    save_imgs = torch.cat([save_imgs, img.unsqueeze(0)], dim=0)


if __name__ == '__main__':
    ckpt_iter = 0
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path)
        iter = ckpt['iter']
        net_D.load_state_dict(ckpt['net_D'])
        net_G.load_state_dict(ckpt['net_G'])
        optimizerD.load_state_dict(ckpt['opti_D'])
        optimizerG.load_state_dict(ckpt['opti_G'])
        save_imgs = ckpt['save_imgs']

    p = None
    iter = 0
    for epoch in range(epochs):
        for i, imgs in tqdm(enumerate(train_loader), total=len(train_loader)):
            iter = len(train_loader) * epoch + i + ckpt_iter
            # 平滑标签
            fake_label = torch.rand((batch_size,), dtype=torch.float, device=device) * 0.1
            real_label = torch.rand((batch_size,), dtype=torch.float, device=device) * 0.1 + 0.9
            imgs = imgs.to(device)
            net_D.zero_grad()
            output = net_D(imgs)
            errD_real = criterion(output, real_label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = net_G(noise)
            output = net_D(fake.detach())
            errD_fake = criterion(output, fake_label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            net_G.zero_grad()
            output = net_D(fake)
            errG = criterion(output, real_label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if iter % 50 == 0:
                tqdm.write('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                           % (epoch, epochs, i, len(train_loader),
                              errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            if visulize and iter % 1000 == 0:
                if p is not None:
                    p.join()
                p = threading.Thread(visulization())
                p.start()
        if save and epoch > 3:
            torch.save({
                'net_G': net_G.state_dict(),
                'net_D': net_D.state_dict(),
                'opti_G': optimizerG.state_dict(),
                'opti_D': optimizerD.state_dict(),
                'iter': iter,
                'save_imgs': save_imgs
            }, 'DCGAN_{}.pth'.format(epoch))
