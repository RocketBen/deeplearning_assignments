import math
import spacy
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from utils import *


class LSTM(nn.Module):
    def __init__(self, en_vocab_size=5967, zh_vocab_size=12016, embed_size=300,
                 use_embed_pretain=True,
                 freeze_embedding=True):
        super(LSTM, self).__init__()
        self.name = 'LSTM'
        self.max_seq_len = max_seq_len
        if use_embed_pretain:
            self.en_embed_layer = nn.Embedding.from_pretrained(get_pretrained_embedding('en'), freeze=freeze_embedding)
            self.zh_embed_layer = nn.Embedding.from_pretrained(get_pretrained_embedding('zh'), freeze=freeze_embedding)
        else:
            self.en_embed_layer = nn.Embedding(en_vocab_size, embed_size)
            self.zh_embed_layer = nn.Embedding(zh_vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, embed_size, bidirectional=False, batch_first=True, num_layers=2,
                               dropout=0.15)
        self.decoder = nn.LSTM(embed_size, embed_size, bidirectional=False, batch_first=True, num_layers=2,
                               dropout=0.15)
        self.fc = nn.Linear(embed_size, zh_vocab_size)
        self.zh_vocab = None
        self.en_spacy = None
        self.en_vocab = None

    def forward(self, en_input, zh_input, teaching=True):
        zh_feed, zh_target = zh_input[:, :-1], zh_input[:, 1:]
        en_embed = self.en_embed_layer(en_input)
        encoder_out, hidden = self.encoder(en_embed)
        zh_embed = self.zh_embed_layer(zh_feed)  # (batch, seq_len, ebd_size)
        fc_outs = []
        decode_out = zh_embed[:, 0, :].unsqueeze(1)
        for i in range(zh_embed.shape[1]):
            if teaching:
                decode_out = zh_embed[:, i, :].unsqueeze(1)
            decode_out, hidden = self.decoder(decode_out, hidden)
            fc_outs.append(self.fc(decode_out))
        log_probs = F.log_softmax(torch.cat(fc_outs, dim=1), dim=2)
        loss = F.nll_loss(log_probs.reshape(log_probs.shape[0] * log_probs.shape[1], -1),
                          zh_target.reshape(zh_target.shape[0] * zh_target.shape[1]),
                          ignore_index=0)
        return log_probs, loss

    def translate(self, en_input):
        if self.zh_vocab is None:
            self.zh_vocab = MTVocab('zh')
        if type(en_input) is str:
            if self.en_spacy is None:
                self.en_spacy = spacy.load('en_core_web_md')
            if self.en_vocab is None:
                self.en_vocab = MTVocab('en')
            en_input = ['<bos>'] + [str(w) for w in self.en_spacy(en_input)] + ['<eos>']
            while len(en_input) < max_seq_len:
                en_input.append('<pad>')
            en_input = [self.en_vocab.get_index(w) for w in en_input]
            en_input = torch.LongTensor(en_input).unsqueeze(0).to(self.fc.weight.device)
        en_embed = self.en_embed_layer(en_input)
        encoder_out, hidden = self.encoder(en_embed)
        decode_out = torch.LongTensor([self.zh_vocab.get_index('<bos>'), ]).unsqueeze(0).to(
            en_input.device)
        decode_out = self.zh_embed_layer(decode_out)
        pred_words, pred_ids = [], []
        while len(pred_words) < max_seq_len - 1:
            decode_out, hidden = self.decoder(decode_out, hidden)
            fc_out = self.fc(decode_out)
            log_prob = F.log_softmax(fc_out, dim=2)
            pred_idx = torch.argmax(log_prob, dim=2).item()
            pred_ids.append(pred_idx)
            pred_word = self.zh_vocab.get_word(pred_idx)
            if pred_word == '<eos>':
                break
            pred_words.append(pred_word)
        return ''.join(pred_words), pred_ids


class MultiHeadSelfAttnLayer(nn.Module):
    def __init__(self, d_model, num_head=1):
        super(MultiHeadSelfAttnLayer, self).__init__()
        # 输入 (batch, seq_len, d_model)
        assert d_model % num_head == 0
        self.num_head = num_head
        self.head_dim = d_model // num_head
        self.W_q = Parameter(torch.randn([d_model, d_model]))
        self.W_k = Parameter(torch.randn((d_model, d_model)))
        self.W_v = Parameter(torch.randn((d_model, d_model)))
        self.W_o = Parameter(torch.randn((d_model, d_model)))
        self.scale_term = math.sqrt(self.head_dim)
        nn.init.kaiming_uniform_(self.W_o)
        nn.init.kaiming_uniform_(self.W_q)
        nn.init.kaiming_uniform_(self.W_k)
        nn.init.kaiming_uniform_(self.W_v)

    def forward(self, x_q, x_k, attn_mask=None):
        # x_q == x_k 时，就是自己跟自己做self-attention
        (batch, q_len, d_model) = x_q.shape
        k_len = x_k.shape[1]
        Q = torch.matmul(x_q, self.W_q)  # (batch, q_len, d_model)
        K = torch.matmul(x_k, self.W_k)  # (batch, k_len, d_model)
        V = torch.matmul(x_k, self.W_v)  # (batch, k_len, d_model)
        # 划分成多个头
        Q = Q.reshape(batch, q_len, self.num_head, self.head_dim)
        K = K.reshape(batch, k_len, self.num_head, self.head_dim)
        V = V.reshape(batch, k_len, self.num_head, self.head_dim)
        # 交换维度方便矩阵相乘
        Q = Q.transpose(1, 2)  # (batch, num_head, q_len, head_dim)
        K = K.transpose(1, 2)  # (batch, num_head, k_len, head_dim)
        V = V.transpose(1, 2)  # (batch, num_head, k_len, head_dim)
        K_T = K.transpose(2, 3)  # (batch, num_head, head_dim, k_len)
        attn_score = torch.matmul(Q, K_T) / self.scale_term
        if attn_mask is not None:
            attn_score += attn_mask
        attn_score = F.softmax(attn_score, dim=-1)  # (batch, num_head, q_len, k_len)
        self.attention_score = attn_score  # 保存方便可视化
        weighted_sum = torch.matmul(attn_score, V)  # (batch, num_head, q_len, head_dim)
        weighted_sum = weighted_sum.transpose(1, 2)  # (batch, q_len, num_head, head_dim)
        weighted_sum = weighted_sum.reshape(batch, q_len, d_model)
        output = torch.matmul(weighted_sum, self.W_o)  # (batch, k_len,d_model)
        output = F.dropout(output, 0.2)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, d_model=300, num_head=1, d_feed=512):
        super(TransformerBlock, self).__init__()
        self.self_attn_layer = MultiHeadSelfAttnLayer(d_model, num_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_feed),
                                          nn.GELU(),
                                          nn.Linear(d_feed, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x_q, x_k):
        attned_x = self.self_attn_layer(x_q, x_k)
        x = x_q + attned_x
        x_norm1 = self.norm1(x)
        x = self.feed_forward(x)
        x += x_norm1
        return x


class Encoder(nn.Module):
    def __init__(self, d_model=300, d_feed=1024, num_head=4, num_layer=3):
        super(Encoder, self).__init__()
        self.tfblocks = nn.ModuleList([
            TransformerBlock(d_model, num_head, d_feed) for i in range(num_layer)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for i in range(num_layer)
        ])

    def forward(self, x):
        for tfblock, norm_layer in zip(self.tfblocks, self.norms):
            block_out = tfblock(x, x.clone())
            x = block_out + x.clone()
            x = norm_layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model=300, d_feed=512, num_head=4, num_layer=3):
        super(Decoder, self).__init__()
        self.self_attn_layers = nn.ModuleList([
            MultiHeadSelfAttnLayer(d_model, num_head) for i in range(num_layer)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for i in range(num_layer)
        ])
        self.tfblocks = nn.ModuleList([
            TransformerBlock(d_model, num_head, d_feed) for i in range(num_layer)
        ])

    def forward(self, decoder_in, encoder_out, decoder_mask=None):
        decoder_out = decoder_in
        if decoder_mask is None:
            decoder_mask = get_attn_mask(decoder_in.shape[1]).to(decoder_in.device)
        for self_attn, norm, tfblock in zip(
                self.self_attn_layers,
                self.norms,
                self.tfblocks
        ):
            attned_decoder_out = self_attn(decoder_out, decoder_out.clone(), attn_mask=decoder_mask)
            decoder_out = decoder_out.clone() + attned_decoder_out
            decoder_out = norm(decoder_out)
            decoder_out = tfblock(decoder_out, encoder_out)
        return decoder_out


class Transformer(nn.Module):
    def __init__(self, en_vocab_size=5967, zh_vocab_size=12016, embed_size=300,
                 use_embed_pretain=True,
                 freeze_embedding=True):
        super(Transformer, self).__init__()
        self.name = 'transformer'
        self.max_seq_len = max_seq_len
        if use_embed_pretain:
            self.en_embed_layer = nn.Embedding.from_pretrained(get_pretrained_embedding('en'), freeze=freeze_embedding)
            self.zh_embed_layer = nn.Embedding.from_pretrained(get_pretrained_embedding('zh'), freeze=freeze_embedding)
        else:
            self.en_embed_layer = nn.Embedding(en_vocab_size, embed_size)
            self.zh_embed_layer = nn.Embedding(zh_vocab_size, embed_size)

        self.encoder = Encoder(embed_size, num_head=3, num_layer=2)
        self.decoder = Decoder(embed_size, num_head=3, num_layer=2)
        position_encoding = get_pos_encoding_matrix(embed_size, max_seq_len)
        decoder_mask = get_attn_mask(max_seq_len - 1)
        self.register_buffer('position_encoding', position_encoding)
        self.register_buffer('decoder_mask', decoder_mask)
        self.fc = nn.Linear(embed_size, zh_vocab_size)
        self.zh_vocab = None
        self.en_spacy = None
        self.en_vocab = None

    def forward(self, en_input, zh_input, teaching):
        zh_feed, zh_target = zh_input[:, :-1], zh_input[:, 1:]
        en_embed = self.en_embed_layer(en_input)
        zh_embed = self.zh_embed_layer(zh_feed)
        en_batch_len = en_input.shape[1]
        zh_batch_len = zh_feed.shape[1]
        en_embed += self.position_encoding[:en_batch_len, :]
        zh_embed += self.position_encoding[:zh_batch_len, :]
        encoder_out = self.encoder(en_embed)
        decoder_mask = self.decoder_mask
        if self.decoder_mask.shape[1] != zh_embed.shape[1]:
            decoder_mask = get_attn_mask(zh_embed.shape[1]).to(zh_embed.device)
        decoder_out = self.decoder(zh_embed, encoder_out, decoder_mask)
        fc_out = self.fc(decoder_out)
        log_probs = F.log_softmax(fc_out, dim=2)
        loss = F.nll_loss(log_probs.reshape(log_probs.shape[0] * log_probs.shape[1], -1),
                          zh_target.reshape(zh_target.shape[0] * zh_target.shape[1]),
                          ignore_index=0)
        return log_probs, loss

    def translate(self, en_input):
        with torch.no_grad():
            if self.zh_vocab is None:
                self.zh_vocab = MTVocab('zh')
            if type(en_input) is str:
                en_input = en_input.lower()
                if self.en_spacy is None:
                    self.en_spacy = spacy.load('en_core_web_md')
                if self.en_vocab is None:
                    self.en_vocab = MTVocab('en')
                en_input = ['<bos>'] + [str(w) for w in self.en_spacy(en_input)] + ['<eos>']
                while len(en_input) < max_seq_len:
                    en_input.append('<pad>')
                en_input = [self.en_vocab.get_index(w) for w in en_input]
                en_input = torch.LongTensor(en_input).unsqueeze(0).to(self.fc.weight.device)
            en_seq_len = en_input.shape[1]
            en_embed = self.en_embed_layer(en_input)
            en_embed += self.position_encoding[: en_seq_len, :]
            encoder_out = self.encoder(en_embed)
            zh_input = torch.LongTensor([self.zh_vocab.get_index('<bos>'), ]).to(en_embed.device).unsqueeze(0)
            zh_sel_len = zh_input.shape[1]
            zh_embed = self.zh_embed_layer(zh_input)
            pred_words = []
            pred_ids = []
            while len(pred_words) < max_seq_len - 1:
                decoder_mask = get_attn_mask(zh_sel_len).to(zh_embed.device)
                zh_embed += self.position_encoding[:zh_sel_len, :]
                decoder_output = self.decoder(zh_embed,
                                              encoder_out,
                                              decoder_mask)
                log_prob = F.log_softmax(self.fc(decoder_output[0, -1, :]), dim=0)
                pred_idx = torch.argmax(log_prob).item()
                pred_ids.append(pred_idx)
                pred_word = self.zh_vocab.get_word(pred_idx)
                if pred_word == '<eos>':  # 预测到句子结尾结束循环
                    break
                pred_words.append(pred_word)
                pred_idx = torch.LongTensor([pred_idx, ]).to(zh_input.device).unsqueeze(0)
                zh_input = torch.cat((zh_input, pred_idx), dim=1)
                zh_embed = self.zh_embed_layer(zh_input)
                zh_sel_len = zh_input.shape[1]
            return ''.join(pred_words), pred_ids


if __name__ == '__main__':
    pass
