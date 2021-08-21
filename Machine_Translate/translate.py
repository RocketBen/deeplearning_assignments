from utils import *
import models
import torch
from torch.utils import data
from matplotlib import pyplot as plt
ckpt_path = '/home/clb/PycharmProjects/deeplearning_assiments/Machine_Translation/transformer_2800_0.5380.pth'

device = torch.device('cuda:3')
model = models.Transformer()
ckpt = torch.load(ckpt_path)
model.load_state_dict(ckpt['model'])
model = model.to(device)


if __name__ == '__main__':
    with torch.no_grad():
        x = 'mathematics is my favorite subject.'
        y = model.translate(x)
        print(y)
        attn = model.decoder.tfblocks[0].self_attn_layer.attention_score.squeeze().cpu().numpy()
        plt.figure()
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.imshow(attn[i, :, :])
        plt.suptitle(x + '\n' + y)
        plt.show()
