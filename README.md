# deeplearning_assignments
2021暑期深度学习课程实验


#  图像分类：CIFAR-10


# 图像生成：CelebA


# 文本分类：IMDB
尝试了多个模型，效果最好的是最简单的[EmebddingBag](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)  
可能是任务比较简单，复杂的模型反而特别容易过拟合  
 | model | RNN | LSTM | Bidirectional-LSTM | Transformer | EmbeddingBag |     
 | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |  
 | accuracy | 74.57 | 83.26 | 84.32 | 80.42 | 87.35 | 

# 机器翻译：CMN-ENG
预处理：给的数据集好像预处理有些错误，使用了spacy重新做分词并按8：1：1重新划train, test, val  
使用了两个模型，分别是基于LSTM的seq2seq，和自己复现的transformer，精度分别为 **31.58%** 和 **51.55%**  
此外，用了spacy的词向量初始化embedding layer  
一些翻译效果展示：  
![image](https://github.com/cenlibin/deeplearning_assignments/blob/main/Images/translation_examples.png)  
绘制了一下attention的可视化（3个head）:   
![image](https://github.com/cenlibin/deeplearning_assignments/blob/main/Images/attention_heatmap.png)  
