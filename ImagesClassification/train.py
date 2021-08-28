import torch
from torch import nn
from torch.nn import functional as F
from utils import *
from torch.utils import data
from models import *
from matplotlib import pyplot as plt
import torchvision.transforms as T
import torchvision
import random

device = torch.device('cuda:3')
epochs = 80
batch_size = 64
lr = 1e-3
ckpt_path = '/home/clb/PycharmProjects/deeplearning_assiments/Images_Classification/ResNet20_59100_0.7328.pth'
save = True
test = True

model = ResNet(3)
model.apply(weights_init)
model.to(device)
opti = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(opti,
                                                 milestones=[30, 60],
                                                 gamma=0.1)
train_loader = data.DataLoader(Cifar10Dataset(is_train=True,
                                              transform=T.Compose([
                                                  T.Resize(36),
                                                  T.RandomResizedCrop(32),
                                                  T.RandomHorizontalFlip(),
                                                  T.ToTensor(),
                                                  T.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])
                                              ])),
                               batch_size,
                               shuffle=True,
                               pin_memory=True,
                               num_workers=4)
test_loader = data.DataLoader(Cifar10Dataset(is_train=False,
                                             transform=T.Compose([
                                                 T.Resize(36),
                                                 T.CenterCrop(32),
                                                 T.ToTensor(),
                                                 T.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])
                                             ])),
                              batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=4)

best_accu = 0
train_iters, test_iters = [], []
train_losses, test_losses, test_accus = [], [], []


def visualization():
    plt.figure(figsize=[15, 7.5])
    # plt.suptitle(model.name, fontsize=20)
    plt.subplot(1, 2, 1)
    plt.ylim(1.5, 2.5)
    plt.plot(train_iters, train_losses, color='red')
    plt.plot(test_iters, test_losses, color='yellow')
    plt.legend(['train_loss', 'test_loss'])
    plt.subplot(1, 2, 2)
    plt.plot(test_iters, test_accus, color='blue')
    plt.legend(['test_accus'])
    plt.show()


if __name__ == '__main__':
    ckpt_iter = 0
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        opti.load_state_dict(ckpt['opti'])
        ckpt_iter = ckpt['iter']
        train_iters = ckpt['train_iters']
        train_losses = ckpt['train_losses']
        test_iters = ckpt['test_iters']
        test_losses = ckpt['test_losses']
        test_accus = ckpt['test_accus']

    print(model)
    postfix_dist = {'train_loss': 0}
    for epoch in range(epochs):
        for i, (imgs, labels) in tqdm(enumerate(train_loader), total=len(train_loader),
                                      desc='epoch: {}'.format(epoch)):
            iter = len(train_loader) * epoch + i + ckpt_iter
            opti.zero_grad()
            imgs, labels = imgs.to(device), labels.to(device)
            pred = model(imgs)
            loss = F.cross_entropy(pred, labels)
            loss.backward()
            opti.step()
            postfix_dist['train_loss'] = loss.cpu().data
            if iter % 50 == 0:
                tqdm.write('iter: {}  train_loss: {}  lr: {}'.format(iter,
                                                                     loss,
                                                                     opti.state_dict()['param_groups'][0]['lr']))
                train_iters.append(iter)
                train_losses.append(loss.item())
            if iter % 1000 == 0 and test:
                accu, test_loss = model_test(model, device=device, test_loader=test_loader)
                tqdm.write('iter: {}  test_accu: {}, test_loss: {} '.format(iter, accu, test_loss))
                test_iters.append(iter)
                test_accus.append(accu)
                test_losses.append(test_loss)
                # 可视化
                visualization()
                if save and accu > 0.63:
                    torch.save({'model': model.state_dict(),
                                'iter': iter,
                                'opti': opti.state_dict(),
                                'train_iters': train_iters,
                                'train_losses': train_losses,
                                'test_iters': test_iters,
                                'test_losses': test_losses,
                                'test_accus': test_accus}, '{}_{}_{:.4f}.pth'.format(model.name, iter, accu))
        scheduler.step()
    accu, test_loss, example = model_test(model, device=device, test_loader=test_loader)
    tqdm.write('iter: {}  test_accu: {}, test_loss: {} '.format(iter, accu, test_loss))
    visualization()
    if save:
        torch.save({'model': model.state_dict(),
                    'iter': iter,
                    'opti': opti.state_dict(),
                    'train_iters': train_iters,
                    'train_losses': train_losses,
                    'test_iters': test_iters,
                    'test_losses': test_losses,
                    'test_accus': test_accus}, '{}_{}_{:.4f}.pth'.format(model.name, 'final', accu))
