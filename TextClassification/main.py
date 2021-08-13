import torch
from torch import nn
from torch.nn import functional as F
from utils import *
from torch.utils import data
import models
from matplotlib import pyplot as plt

device = torch.device('cuda:1')
epochs = 50
batch_size = 32
lr = 5e-5
ckpt_path = None

model = models.TransformerClassifier(pretrained_embedding=True)
model_name = 'SelfAttention'
opti = torch.optim.Adam(model.parameters(), lr=lr)

train_set = IMDBDataset(is_train=True, device=None)
test_set = IMDBDataset(is_train=False, device=None)
train_loader = data.DataLoader(train_set, batch_size, collate_fn=train_set.collate_fn, shuffle=True)
test_loader = data.DataLoader(test_set, batch_size, collate_fn=test_set.collate_fn, shuffle=True)

best_accu = 0
train_iters, test_iters = [], []
train_losses, test_losses, test_accus = [], [], []


def visualization():
    plt.figure(figsize=[15, 7.5])
    plt.subplot(1, 2, 1)
    plt.ylim(0, 1.5)
    plt.plot(train_iters, train_losses, color='red')
    plt.plot(test_iters, test_losses, color='yellow')
    plt.subplot(1, 2, 2)
    plt.plot(test_iters, test_accus, color='blue')
    plt.show()


if __name__ == '__main__':
    ckpt_iter = 0
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        opti.load_state_dict(ckpt['opti'])
        ckpt_iter = ckpt['iter']
        train_iters = ckpt['train_iters']
        train_losses = ckpt['train_losses']
        test_iters = ckpt['test_iters']
        test_losses = ckpt['test_losses']
        test_accus = ckpt['test_accus']
    model = model.to(device)
    print(model)
    postfix_dist = {'train_loss': 0}
    for epoch in range(epochs):
        for i, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader),
                              desc='epoch: {}'.format(epoch)):
            iter = len(train_loader) * epoch + i + ckpt_iter
            x, y = x.to(device), y.to(device)
            pred = model(x)
            opti.zero_grad()
            loss = F.nll_loss(torch.log(pred), y)
            loss.backward()
            opti.step()
            postfix_dist['train_loss'] = loss.cpu().data
            if iter % 50 == 0:
                tqdm.write('iter: {}  train_loss: {}'.format(iter, loss))
                train_iters.append(iter)
                train_losses.append(loss.cpu().data)
            if iter % 500 == 0:
                accu, test_loss = model_test(model, device=device, test_len=100, test_loader=test_loader)
                tqdm.write('iter: {}  test_accu: {}, test_loss: {} '.format(iter, accu, test_loss))
                test_iters.append(iter)
                test_accus.append(accu)
                test_losses.append(test_loss.cpu().data)
                # 可视化
                visualization()
                torch.save({'model': model.state_dict(),
                            'iter': iter,
                            'opti': opti.state_dict(),
                            'train_iters': train_iters,
                            'train_losses': train_losses,
                            'test_iters': test_iters,
                            'test_losses': test_loss,
                            'test_accus': test_accus}, '{}_{}_{:.4f}.pth'.format(model_name, iter, accu))
    accu, test_loss = model_test(model, device=device, test_loader=test_loader)
    tqdm.write('iter: {}  test_accu: {}, test_loss: {} '.format(iter, accu, test_loss))
    visualization()
    torch.save({'model': model.state_dict(),
                'iter': iter,
                'opti': opti.state_dict(),
                'train_iters': train_iters,
                'train_losses': train_losses,
                'test_iters': test_iters,
                'test_losses': test_loss,
                'test_accus': test_accus}, '{}_{}_{:.4f}.pth'.format(model_name, 'final', accu))
