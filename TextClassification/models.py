import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from utils import *


class RnnClassifier(nn.Module):
    def __init__(self):
        super(RnnClassifier, self).__init__()
        self.name = 'Rnn'
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(freq_vocab_size, embedding_size)
        self.rnn = nn.RNN(embedding_size, 1, bidirectional=False, batch_first=True)
        self.fc1 = nn.Linear(self.max_seq_len, 2)

    def forward(self, x0):
        x0 = self.embedding(x0)
        x0 = torch.tanh(x0)
        x0 = F.dropout(x0, 0.5)
        x1, _ = self.rnn(x0)
        x1 = F.dropout(x1, 0.1)
        x2 = x1.reshape(x1.shape[0], -1)
        x3 = self.fc1(x2)
        x3 = F.softmax(x3, -1)
        return x3


class LstmClassifier(nn.Module):
    def __init__(self):
        super(LstmClassifier, self).__init__()
        self.name = 'LSTM'
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(freq_vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, 1, bidirectional=False, batch_first=True)
        self.fc1 = nn.Linear(self.max_seq_len, 2)

    def forward(self, x0):
        x0 = self.embedding(x0)
        x0 = torch.tanh(x0)
        x0 = F.dropout(x0, 0.5)
        x1, _ = self.lstm(x0)
        x1 = F.dropout(x1, 0.1)
        x2 = x1.reshape(x1.shape[0], -1)
        x3 = self.fc1(x2)
        x3 = F.softmax(x3, -1)
        return x3


class BiLstmClassifier(nn.Module):
    def __init__(self):
        super(BiLstmClassifier, self).__init__()
        self.name = 'BiLSTM'
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(freq_vocab_size, embedding_size)
        self.bi_lstm = nn.LSTM(embedding_size, 1, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(self.max_seq_len * 2, 2)

    def forward(self, x0):
        x0 = self.embedding(x0)
        x0 = torch.tanh(x0)
        x0 = F.dropout(x0, 0.5)
        x1, _ = self.bi_lstm(x0)
        x1 = F.dropout(x1, 0.1)
        x2 = x1.reshape(x1.shape[0], -1)
        x3 = self.fc1(x2)
        x3 = F.softmax(x3, -1)
        return x3


class TransformerClassifier(nn.Module):
    def __init__(self):
        super(TransformerClassifier, self).__init__()
        self.name = 'Transformer'
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(freq_vocab_size, embedding_size)
        self.self_attention_layer = nn.TransformerEncoderLayer(embedding_size, 1, batch_first=True, dim_feedforward=512)
        self.encoder = nn.TransformerEncoder(self.self_attention_layer, num_layers=1)
        self.position_encoding = get_pos_embed_matrix(embedding_size, max_seq_len).T
        self.fc = nn.Linear(self.max_seq_len * embedding_size, 2)

    def forward(self, x0):
        x1 = self.embedding(x0)
        x2 = torch.tanh(x1)
        x2 = F.dropout(x2, 0.5)
        self.position_encoding = self.position_encoding.to(x2.device)
        x3 = self.position_encoding + x2
        x4 = self.encoder(x3)
        x4 = x4.reshape(x4.shape[0], -1)
        x5 = self.fc(x4)
        x6 = torch.softmax(x5, -1)
        return x6


class EmbeddingBagClassifier(nn.Module):
    def __init__(self):
        super(EmbeddingBagClassifier, self).__init__()
        self.name = 'EmbeddingBag'
        self.embedding_bag = nn.EmbeddingBag(freq_vocab_size, embedding_size, mode='mean')
        self.fc = nn.Linear(embedding_size, 2)

    def forward(self, x):
        x = self.embedding_bag(x)
        x = torch.tanh(x)
        x = F.dropout(x, 0.5)
        x = self.fc(x)
        x = torch.softmax(x, -1)
        return x


if __name__ == '__main__':
    pass
