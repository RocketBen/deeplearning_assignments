from utils import *
import models
import torch
from torch.utils import data
from matplotlib import pyplot as plt

device = torch.device('cuda:2')

ckpt_path = '/home/clb/PycharmProjects/deeplearning_assiments/Machine_Translation/transformer_11000_0.5236.pth'
transformer = models.Transformer()
ckpt = torch.load(ckpt_path)
transformer.load_state_dict(ckpt['model'])
transformer = transformer.to(device)
ckpt_path = '/home/clb/PycharmProjects/deeplearning_assiments/Machine_Translation/LSTM_8000_0.3259.pth'
lstm = models.LSTM()
ckpt = torch.load(ckpt_path)
lstm.load_state_dict(ckpt['model'])
lstm = lstm.to(device)
vis = False
if __name__ == '__main__':
    with torch.no_grad():
        data = []
        with open('./CMN-ENG/cmn.txt') as f:
            for line in f:
                en, zh, _ = line.strip().split('\t')
                data.append((en, zh))
        random.shuffle(data)
        for en, zh in data[:10]:
            print('input: ', en)
            print('label: ', zh)
            print('lstm: ', lstm.translate(en)[0])
            print('transformer: ', transformer.translate(en)[0])
            if vis:
                attn = transformer.decoder.tfblocks[0].self_attn_layer.attention_score.cpu().numpy()[0]
                plt.figure()
                plt.suptitle(en)
                for i in range(attn.shape[0]):
                    plt.subplot(1, attn.shape[0], i + 1)
                    plt.imshow(attn[i])
                plt.show()
