import os
from os.path import join
import random
from collections import Counter
import spacy
import pandas as pd
import numpy as np
from utils import *

root = './CMN-ENG'


def tokenization():
    # 分词
    spacy_en = spacy.load('en_core_web_md')
    spacy_zh = spacy.load('zh_core_web_md')
    lines = []
    with open(join(root, 'cmn.txt'), 'r') as f:
        for line in f:
            en, zh, _ = line.strip().split('\t')
            lines.append(''.join([str(w) + ' ' for w in spacy_en(en)]) + '\t' + ''.join(
                [str(w) + ' ' for w in spacy_zh(zh)]) + '\n')
            print(lines[-1])
    with open(join(root, 'tokenized.txt'), 'w') as f:
        f.writelines(lines)


def split_data():
    # 划分 train test val set
    ratio = [.7, .2, .1]
    data = []
    with open(join(root, 'tokenized.txt'), 'r') as f:
        data = [line for line in f]
    random.shuffle(data)
    pre_idx = 0
    for i, set_name in enumerate(['train', 'test', 'val']):
        set_len = int(ratio[i] * len(data))
        set_data = data[pre_idx: pre_idx + set_len]
        pre_idx += set_len
        with open(join(root, set_name + '.txt'), 'w') as f:
            f.writelines(set_data)


def make_vocab():
    # 根据训练集词频划分词典
    ens, zhs = [], []
    with open(join(root, 'train.txt'), 'r') as f:
        for line in f:
            line = line.lower()
            en, zh = line.strip('\n').split('\t')
            en, zh = en.split(), zh.split()
            ens += en
            zhs += zh
    en_cnt, zh_cnt = Counter(ens), Counter(zhs)
    special_token = ['<pad>', '<unk>', '<bos>', '<eos>']
    en_vocab = special_token + [word for (word, cnt) in en_cnt.most_common(10000)]
    zh_vocab = special_token + [word for (word, cnt) in zh_cnt.most_common(20000)]
    for filename, vocab in [('en.vocab', en_vocab), ('zh.vocab', zh_vocab)]:
        with open(join(root, filename), 'w') as f:
            for w in vocab:
                f.write(w + '\n')


def pad_and_save_index():
    # 填充到固定长度并保存成字典序号
    max_sentence_len = 50  # 对英语和中文都统一长度到50
    indexed_root = join(root, 'indexed')
    try:
        os.makedirs(indexed_root)
    except FileExistsError:
        pass
    en_vocab, zh_vocab = MTVocab('en'), MTVocab('zh')
    for i, set_name in enumerate(['train', 'test', 'val']):
        df = []
        with open(join(root, set_name + '.txt'), 'r') as rf:
            for line in rf:
                en, zh = line.strip('\n').split('\t')
                en, zh = en.split(), zh.split()
                en = ['<bos>', ] + en + ['<eos>', ]
                while len(en) < max_sentence_len:
                    en.append('<pad>')
                zh = ['<bos>', ] + zh + ['<eos>', ]
                while len(zh) < max_sentence_len:
                    zh.append('<pad>')
                en = [en_vocab.get_index(w) for w in en]
                zh = [zh_vocab.get_index(w) for w in zh]
                df.append(en + zh)
        df = pd.DataFrame(df)
        df.to_csv(join(indexed_root, set_name + '.csv'), header=False, index=False, sep=',')


if __name__ == '__main__':
    make_vocab()
    pad_and_save_index()
