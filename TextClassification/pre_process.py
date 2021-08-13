"""
a pre-process script for IMDB dataset
tokenize and vectorize the sentence using spacy and combine multiple txt and its label as a single csv file
"""
import os
from os.path import join
import spacy
import pandas as pd
from utils import *
from tqdm import tqdm
import random
import numpy as np


def tokenization():
    spacy.prefer_gpu()
    nlp = spacy.load('en_core_web_md')
    target_path = './IMDB/tokenized'
    try:
        os.makedirs(target_path)
    except FileExistsError:
        pass
    for t in ['test', 'train']:
        data_frame = []
        for label_id, label in enumerate(['pos', 'neg']):
            src_path = './IMDB/{}/{}'.format(t, label)
            print(src_path)
            for _, _, files in os.walk(src_path):
                for file in tqdm(files):
                    file_path = join(src_path, file)
                    with open(file_path, 'rb') as f:
                        text = f.readline().decode('utf-8')
                        words = nlp(text)
                        words = [str(w) for w in words]
                        indexs = ['<bos>']
                        indexs.extend([w for w in words])
                        indexs.append('<eos>')
                        while len(indexs) < max_seq_len:
                            indexs.append('<pad>')
                        indexs.append(label)
                        data_frame.append(indexs)
        random.shuffle(data_frame)
        data_frame = pd.DataFrame(data_frame)
        data_frame.to_csv(target_path + '/{}.csv'.format(t), index=False, sep=',')


def save_index():
    vocab = ImdbVocab(word_cnt=max_word_cnt)
    target_path = './IMDB/indexed'
    try:
        os.makedirs(target_path)
    except FileExistsError:
        pass
    for t in ['test.csv', 'train.csv']:
        src_path = './IMDB/tokenized/{}'.format(t)
        data_frame = pd.read_csv(src_path).values
        target_frame = np.empty_like(data_frame, dtype=np.int64)
        for i in tqdm(range(data_frame.shape[0])):
            for j in range(data_frame.shape[1] - 1):
                target_frame[i, j] = vocab.get_index(str(data_frame[i, j]))
            label = data_frame[i, -1]
            target_frame[i, -1] = 1 if str(label) == 'pos' else 0
        target_frame = pd.DataFrame(target_frame)
        target_frame.to_csv(join(target_path, t), header=False, index=False, sep=',')


def split_csv():
    target_path = './IMDB/tokenized'
    for t in ['test', 'train']:
        try:
            os.makedirs(join(target_path, t))
        except FileExistsError:
            pass
        data_frame = pd.read_csv(join(target_path, t + '.csv'))
        for i, row in data_frame.iterrows():
            row.T.to_csv(join(target_path, t, str(i) + '.csv'), index=False, sep=',')


def vectorization():
    src_path = './IMDB/tokenized'
    target_path = './IMDB/vectorized'
    nlp = spacy.load('en_core_web_lg')
    vocab = ImdbVocab()
    for t in ['test', 'train']:
        try:
            os.makedirs(join(target_path, t))
        except FileExistsError:
            pass
        data_frame = pd.read_csv(join(src_path, t + '.csv'))
        print(target_path)
        for i, row in tqdm(data_frame.iterrows(), total=25000):
            np_array = row.values
            text_index, label_id = np_array[:-1], np_array[-1]
            vectors = np.zeros([text_index.shape[0], 300])
            for j, word_index in enumerate(text_index):
                word = vocab.get_word(word_index)
                word_vec = nlp.vocab[word].vector
                vectors[j, :] = word_vec
            seq_df = pd.DataFrame(vectors)
            seq_df.to_csv(join(target_path, t, '{}.{}'.format(i, label_id)), header=False, index=False, sep=',')


def words_freq_count():
    # 生成词频统计数据，方便构建常用词词典
    src_vocab = ImdbVocab()
    cnt = np.zeros(len(src_vocab), dtype=np.int64)
    target_path = './IMDB/tokenized'
    for t in ['train']:
        data_frame = pd.read_csv(os.path.join(target_path, t + '.csv'))
        for i, row in data_frame.iterrows():
            for word in row.values[:-1]:
                cnt[src_vocab.get_index(str(word))] += 1
    sorted_idx = np.argsort(cnt)[::-1]  # 按词频排序
    str_arr = np.array([src_vocab.get_word(w) for w in range(len(src_vocab))], dtype=np.str)
    cnt = np.stack([np.arange(len(src_vocab), dtype=np.int64), cnt])
    cnt = cnt[:, sorted_idx]
    str_arr = str_arr[sorted_idx]
    cnt = cnt[:, 4:]  # 去除特殊字符统计
    str_arr = str_arr[4:]
    # 保存成csv方便后续读取
    df = pd.DataFrame()
    df['src_id'] = cnt[0]
    df['freq'] = cnt[1]
    df['words'] = str_arr
    df.to_csv('./IMDB/words_freq_cnt.csv', index=False)


if __name__ == '__main__':
    save_index()
