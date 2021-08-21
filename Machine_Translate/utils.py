from torch.utils import data
import os
import random
import string
import torch
from tqdm import tqdm
from torch.nn import functional as F
import spacy
import numpy as np
from time import time
from torch import nn
import math
import pandas as pd

max_seq_len = 50


class MTDataset(data.Dataset):
    def __init__(self, dataset_type, device=None, force_pad=True):
        super().__init__()
        self.force_pad = force_pad
        device = torch.device('cpu') if device is None else device
        df = pd.read_csv('./CMN-ENG/indexed/{}.csv'.format(dataset_type))
        self.data_list = df.values
        self.data_list = torch.from_numpy(self.data_list).to(device)

    def __getitem__(self, index):
        line = self.data_list[index]
        en, zh = line[:50], line[50:]
        if not self.force_pad:  # 到第一个pad处截断
            en = en[:torch.where(en == 0)[0][0]]
            zh = zh[:torch.where(zh == 0)[0][0]]
        return en, zh

    def __len__(self):
        return self.data_list.shape[0]

    def collate_fn(self, examples):
        ens, zhs = [], []

        max_en_len = max_zh_len = 0
        for (en, zh) in examples:
            ens.append(en)
            zhs.append(zh)
            if not self.force_pad:
                max_en_len = max(max_en_len, en.shape[0])
                max_zh_len = max(max_zh_len, zh.shape[0])

        if not self.force_pad:
            ens = [torch.cat([en, torch.zeros(max_en_len - en.shape[0], dtype=torch.long).to(en.device)], 0) for en in
                   ens]
            zhs = [torch.cat([zh, torch.zeros(max_zh_len - zh.shape[0], dtype=torch.long).to(zh.device)], 0) for zh in
                   zhs]
        ens = torch.stack(ens, dim=0)
        zhs = torch.stack(zhs, dim=0)
        return ens, zhs


class MTVocab:
    def __init__(self, language='en'):
        self._word2index = {}  # 构建字典
        self._index2word = []
        with open('./CMN-ENG/{}.vocab'.format(language)) as f:
            for word in f:
                word = word.strip('\n')
                self._word2index[word] = len(self._word2index)
                self._index2word.append(word)

    def get_index(self, word):
        word = word.lower()
        try:
            index = self._word2index[word]
        except KeyError:
            index = self._word2index['<unk>']
        return index

    def get_word(self, index):
        try:
            word = self._index2word[index]
        except IndexError:
            word = '<unk>'
        return word

    def __len__(self):
        return len(self._index2word)


def get_pos_encoding_matrix(embedding_size, seq_len):  # 生成transformer的位置编码
    res = np.empty([embedding_size, seq_len])
    for pos in range(seq_len):
        for i in range(embedding_size):
            div_term = np.power(10000, 2 * (i // 2) / embedding_size)
            res[i, pos] = np.sin(pos / div_term) if i % 2 == 0 else np.cos(pos / div_term)
    return torch.from_numpy(res).type(torch.FloatTensor).T


def get_pretrained_embedding(language='en'):  # 使用spacy的词向量构建嵌入矩阵
    vocab = MTVocab(language)
    weight = np.zeros([300, len(vocab)])
    nlp = spacy.load('{}_core_web_md'.format(language))
    for i in range(len(vocab)):
        weight[:, i] = nlp.vocab[vocab.get_word(i)].vector
    weight = torch.from_numpy(weight).float()
    return weight.T


def get_attn_mask(seq_len):  # 构建用于transformer解码器的注意力掩码矩阵
    attn_mask = torch.triu(torch.ones((seq_len, seq_len)) == 1).transpose(0, 1)
    attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf'))
    attn_mask = attn_mask.masked_fill(attn_mask == 1, float(0.0))
    return attn_mask


def model_test(model, device, test_loader, test_len=None):
    cnt = 0
    total = 0
    ave_loss = 0
    zh_vocab = MTVocab('zh')
    en_vocab = MTVocab('en')
    if not test_len:
        test_len = len(test_loader)
    else:
        test_len = min(test_len, len(test_loader))

    with torch.no_grad():
        for i, (test_en, test_zh) in tqdm(enumerate(test_loader), total=test_len, desc='test', leave=False):
            if i == test_len:
                break
            test_en, test_zh = test_en.to(device), test_zh.to(device)
            log_prob, loss = model(test_en, test_zh, teaching=False)
            if i == 0:  # 每次测试打印一些翻译样本
                example_en = test_en[0]
                example_en = ''.join(
                    en_vocab.get_word(int(i)) + ' ' for i in example_en if en_vocab.get_word(int(i)) not in ['<pad>', '<bos>', '<eos>'])
                example_zh = log_prob[0]
                example_zh = torch.argmax(example_zh, dim=1)
                example_zh = ''.join(
                    zh_vocab.get_word(int(i)) for i in example_zh if zh_vocab.get_word(int(i)) not in ['<pad>', ])
                example_val, _ = model.translate(test_en[0].unsqueeze(0))
                example = (example_en, example_zh, example_val)
            log_prob = log_prob.reshape(log_prob.shape[0] * log_prob.shape[1], -1)
            test_zh = test_zh[:, 1:]
            test_zh = test_zh.reshape(test_zh.shape[0] * test_zh.shape[1])
            # 计算accuarcy时忽略pad
            total_count = test_zh != 0
            right_classfi = torch.argmax(log_prob, dim=1) == test_zh
            cnt += torch.sum(total_count * right_classfi).item()
            total += torch.sum(total_count).item()
            ave_loss += loss.item()

    return cnt / total, ave_loss / test_len, example


if __name__ == '__main__':
    mask = get_attn_mask(5)
    pe = get_pos_encoding_matrix(300, 50)
