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

max_seq_len = 500  # 最长为2800， 小于2800则截断
max_word_cnt = 11149  # 选择词频在25以上的单词
embedding_size = 300
freq_vocab_size = max_word_cnt + 4


class IMDBDataset(data.Dataset):
    def __init__(self, root=r'./IMDB', is_train=True, device=None):
        super().__init__()
        self.vocab = ImdbVocab(word_cnt=max_word_cnt)
        # 从csv文本中读取所有分好词的样本和标签
        t0 = time()
        df_path = os.path.join(root, 'indexed/{}'.format('train.csv' if is_train else 'test.csv'))
        df = pd.read_csv(df_path)
        self.sentence_list = df.values[:, :max_seq_len]
        self.label_list = df.values[:, -1]
        # np数组转tensor
        self.sentence_list = torch.from_numpy(self.sentence_list)
        self.label_list = torch.from_numpy(self.label_list)

        if device is not None:
            self.sentence_list = self.sentence_list.to(device)
            self.label_list = self.label_list.to(device)
        print('{:.4f}s to load {}'.format(time() - t0, df_path))

    def __getitem__(self, index):
        return self.sentence_list[index], self.label_list[index]

    def __len__(self):
        return self.label_list.shape[0]

    @staticmethod
    def collate_fn(examples):
        indexs, labels = [], []
        for e in examples:
            indexs.append(e[0])
            labels.append(e[1])
        indexs = torch.stack(indexs, dim=0)  # (batch, seq_len, embedding_size)
        labels = torch.stack(labels, dim=0)  # (batch)
        return indexs, labels


def get_pos_embed_matrix(embedding_size, seq_len):  # 生产transformer的位置编码矩阵
    res = np.empty([embedding_size, seq_len])
    for pos in range(seq_len):
        for i in range(embedding_size):
            div_term = np.power(10000, 2 * (i // 2) / embedding_size)
            res[i, pos] = np.sin(pos / div_term) if i % 2 == 0 else np.cos(pos / div_term)
    return torch.from_numpy(res).type(torch.FloatTensor)


class ImdbVocab:
    def __init__(self, word_cnt=None):
        self._word2index = {}  # 构建字典
        self._index2word = []
        for token in ['<pad>', '<unk>', '<bos>', '<eos>']:
            self._word2index[token] = len(self._word2index)  # vocab_size: 89531
            self._index2word.append(token)
        if word_cnt is None:
            with open(os.path.join('./IMDB/imdb.vocab'), 'rb') as f:
                words = f.readlines()
                for i, word in enumerate(words):
                    word = word.decode('utf-8').replace('\n', '').lower()
                    self._word2index[word] = len(self._word2index)
                    self._index2word.append(word)
        else:
            df = pd.read_csv('./IMDB/words_freq_cnt.csv')
            words_list = df['words'][:word_cnt]
            for i, word in enumerate(words_list):
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
            index = self._index2word[index]
        except IndexError:
            index = -1
        return index

    def __len__(self):
        return len(self._index2word)


def model_test(model, device, test_loader, test_len=None):
    cnt = 0
    ave_loss = 0
    if not test_len:
        test_len = len(test_loader)
    with torch.no_grad():
        for i, (test_x, test_y) in tqdm(enumerate(test_loader), total=test_len, desc='test', leave=False):
            if i == test_len:
                break
            test_x, test_y = test_x.to(device), test_y.to(device)
            pred = model(test_x)
            cnt += int(torch.sum(torch.argmax(pred, dim=1) == test_y))
            ave_loss += F.nll_loss(torch.log(pred), test_y).data

    total = test_len * test_loader.batch_size
    return cnt / total, ave_loss / test_len


def get_pretrained_embedding():
    freq_vocab = ImdbVocab(word_cnt=max_word_cnt)
    weight = np.zeros([embedding_size, len(freq_vocab)])
    nlp = spacy.load('en_core_web_md')
    for i in range(len(freq_vocab)):
        weight[:, i] = nlp.vocab[freq_vocab.get_word(i)].vector
    weight = torch.from_numpy(weight).float()

    return weight.T


if __name__ == '__main__':
    pe = get_pos_embed_matrix(10, 15)
