# deeplearning_assignments
2021暑期深度学习课程实验

# TO DO:

## CV:

### 1. 图像分类：CIFAR-10（50,000训练，10,000测试，十分类）
* 7.11 简单搭建了一个16层卷积+4层全连接分类器的网络，在lr=0.001，batch_size为64的情况下训练100个epochs精度57.35%

### 2. 图像生成：CelebA（202,599张人脸图像）

## NLP：

### 1. 文本分类：IMDB（25,000训练，25,000测试，二分类）

### 2. 机器翻译：CMN-ENG（18,000训练，500验证，2636测试，英文到中文的翻译数据集）
预处理：给的数据集好像预处理部分有一些错误，所以我使用spacy重新进行分词并按8：1：1重新划分训练，测试和验证集。  
使用了两个模型，分别是基于LSTM的seq2seq，和自己复现的transformer，验证集精度分别能达到31.58%和51.55%  
可能是数据集比较简单，两个模型编码器和解码器部分都只用了两层，增加层数反而效果更差  
decoder做翻译的时候用最简单的贪心选词，下面是一些效果展示  
![image](deeplearning_assignments/Images/translation_examples.png)  
绘制了一下attention的可视化（三个头）  
![image](deeplearning_assignments/Images/attention_heatmap.png)  
