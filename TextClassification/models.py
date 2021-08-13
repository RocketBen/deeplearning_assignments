import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from utils import *


class RNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNN, self).__init__()
        self.W_h = Parameter(torch.randn([output_size, output_size]))
        self.W_x = Parameter(torch.randn([input_size, output_size]))  # x:(batch_size, seq_len, input_size)
        # 这里实现的pytorch版本与吴恩达课程的RNN略有不同，只用了两个权重矩阵，最后一个hidden_state为最后一个输出y
        self.b_h = Parameter(torch.randn(output_size))
        self.b_x = Parameter(torch.randn(output_size))
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, input, h0=None):
        """

        :param input:  shape of (batch, seq_len, input_size)
        :param h0: shape of (batch, output_size), and will be zero if not provided
        :return:
        y: shape of (batch, seq_len, output_size)
        h: shape of (bacth, output_size)
        """
        (batch_size, seq_len, input_size) = input.shape
        if h0 is None:
            h0 = torch.zeros([batch_size, self.output_size]).to(input.device)
        h = h0
        y = torch.empty([batch_size, seq_len, self.output_size]).to(input.device)
        for t in range(seq_len):
            z_t = torch.mm(h, self.W_h) + self.b_h + torch.mm(input[:, t, :], self.W_x) + self.b_x
            y_t = torch.tanh(z_t)
            y[:, t, :] = y_t
            h = y_t
        return y, h


class SelfAttnLayer(nn.Module):
    def __init__(self, input_size, output_size=None):
        super(SelfAttnLayer, self).__init__()
        # 输入 (batch, seq_len, input_size)
        output_size = input_size if output_size is None else output_size
        self.Q_weight = Parameter(torch.randn([input_size, input_size]))
        self.K_weight = Parameter(torch.randn((input_size, input_size)))
        self.V_weight = Parameter(torch.randn((input_size, output_size)))
        self.scale_term = math.sqrt(input_size)

    def forward(self, x):
        # 输入 (batch, seq_len, input_size)
        Q = x @ self.Q_weight
        K = x @ self.K_weight
        V = x @ self.V_weight  # (batch, seq_len, output_size)
        K_T = K.transpose(1, 2)
        attn_score = F.softmax(Q @ K_T / self.scale_term, dim=2)  # (batch, seq_len, seq_len)
        # 注意力得分左乘， 每行对于一个元素， softmax应在列方向
        weighted_sum = attn_score @ V
        return weighted_sum


class BiLstmClassifier(nn.Module):
    def __init__(self, pretrained_embedding=False):
        super(BiLstmClassifier, self).__init__()
        self.max_seq_len = max_seq_len
        if pretrained_embedding:
            self.embedding = nn.Embedding.from_pretrained(get_pretrained_embedding(), freeze=False)
        else:
            self.embedding = nn.Embedding(freq_vocab_size, embedding_size)
        self.bi_lstm1 = nn.LSTM(embedding_size, embedding_size, bidirectional=True, batch_first=True)
        self.bi_lstm2 = nn.LSTM(embedding_size, 1, bidirectional=True, batch_first=True)
        self.drop_out = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(self.max_seq_len * 2, 2)

    def forward(self, x0):
        x0 = self.embedding(x0)
        x0 = torch.tanh(x0)
        x1, _ = self.bi_lstm1(x0)
        x1 = x1.view(x1.shape[0], x1.shape[1], 2, embedding_size)
        x2 = torch.mean(x1, dim=2, keepdim=False)
        x3 = x2 + x0
        x4 = x3 / 2
        x4 = F.layer_norm(x4, x4.shape[1:])
        x5, _ = self.bi_lstm2(x4)
        x5 = x5.reshape(x5.shape[0], -1)
        x5 = self.drop_out(x5)
        x5 = self.fc1(x5)
        x5 = F.softmax(x5, -1)
        return x5


class RnnClassifier(nn.Module):
    def __init__(self, pretrained_embedding=False):
        super(RnnClassifier, self).__init__()
        self.max_seq_len = max_seq_len
        if pretrained_embedding:
            self.embedding = nn.Embedding.from_pretrained(get_pretrained_embedding(), freeze=True)
        else:
            self.embedding = nn.Embedding(freq_vocab_size, embedding_size)
        self.rnn = nn.RNN(embedding_size, 2, bidirectional=False, batch_first=True)
        self.fc = nn.Linear(self.max_seq_len * 2, 2)

    def forward(self, x):
        x = self.embedding(x)
        x = F.relu(x)
        x, _ = self.rnn(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = F.softmax(x, -1)
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, pretrained_embedding=False):
        super(TransformerClassifier, self).__init__()
        self.max_seq_len = max_seq_len
        if pretrained_embedding:
            self.embedding = nn.Embedding.from_pretrained(get_pretrained_embedding(), freeze=False)
        else:
            self.embedding = nn.Embedding(freq_vocab_size, embedding_size)
        self.self_attention_layer = nn.TransformerEncoderLayer(embedding_size, 5, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.self_attention_layer, 3)
        self.position_encoding = get_pos_embed_matrix(embedding_size, max_seq_len).T
        self.fc = nn.Linear(self.max_seq_len * embedding_size, 2)

    def forward(self, x0):
        x1 = self.embedding(x0)
        x2 = F.tanh(x1)
        self.position_encoding = self.position_encoding.to(x2.device)
        x3 = self.position_encoding + x2
        x4 = self.encoder(x3)
        x4 = x4.reshape(x4.shape[0], -1)
        x4 = F.relu(x4)
        x5 = self.fc(x4)
        x6 = F.softmax(x5, -1)
        return x6


if __name__ == '__main__':
    embedding_size = 64
    seq_len = 2790
    batch_size = 8
    attn_layer = SelfAttnLayer(embedding_size, 2)
    x = torch.randn(batch_size, seq_len, embedding_size)
    y = attn_layer(x)
