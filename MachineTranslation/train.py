import torch
from torch import nn
from torch.nn import functional as F
from utils import *
from torch.utils import data
import models
from matplotlib import pyplot as plt
import random

device = torch.device('cuda:2')
epochs = 80
batch_size = 4
lr = 1e-5
ckpt_path = '/home/clb/PycharmProjects/deeplearning_assiments/Machine_Translation/transformer_11000_0.5236.pth'
save = True
force_pad = False

model = models.Transformer(use_embed_pretain=True, freeze_embedding=False)
model.to(device)
opti = torch.optim.Adam(model.parameters(), lr=lr)

train_set = MTDataset('train', device, force_pad=force_pad)
test_set = MTDataset('test', device, force_pad=force_pad)
train_loader = data.DataLoader(train_set, batch_size, collate_fn=train_set.collate_fn, shuffle=True)
test_loader = data.DataLoader(test_set, batch_size, collate_fn=test_set.collate_fn, shuffle=True)

best_accu = 0
train_iters, test_iters = [], []
train_losses, test_losses, test_accus = [], [], []


def visualization():
    plt.figure(figsize=[15, 7.5])
    plt.suptitle(model.name, fontsize=20)
    plt.subplot(1, 2, 1)
    plt.ylim(0, 7)
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
        for i, (en, zh) in tqdm(enumerate(train_loader), total=len(train_loader),
                                desc='epoch: {}'.format(epoch)):
            iter = len(train_loader) * epoch + i + ckpt_iter
            opti.zero_grad()
            en, zh = en.to(device), zh.to(device)
            log_probs, loss = model(en, zh, teaching=random.random() > 0.5)
            loss.backward()
            opti.step()
            postfix_dist['train_loss'] = loss.cpu().data
            if iter % 50 == 0:
                tqdm.write('iter: {}  train_loss: {}'.format(iter, loss))
                train_iters.append(iter)
                train_losses.append(loss.item())
            if iter % 2000 == 0:
                accu, test_loss, example = model_test(model, device=device, test_loader=test_loader)
                tqdm.write('iter: {}  test_accu: {}, test_loss: {} '.format(iter, accu, test_loss))
                tqdm.write(example[0] + '\n' + example[1] + '\n' + example[2])
                test_iters.append(iter)
                test_accus.append(accu)
                test_losses.append(test_loss)
                # 可视化
                visualization()
                if save and accu > 0.2:
                    torch.save({'model': model.state_dict(),
                                'iter': iter,
                                'opti': opti.state_dict(),
                                'train_iters': train_iters,
                                'train_losses': train_losses,
                                'test_iters': test_iters,
                                'test_losses': test_losses,
                                'test_accus': test_accus}, '{}_{}_{:.4f}.pth'.format(model.name, iter, accu))
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
