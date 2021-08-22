# deeplearning_assignments
2021暑期深度学习课程实验


#  图像分类：CIFAR-10（50,000训练，10,000测试，十分类）


# 图像生成：CelebA（202,599张人脸图像）


# 文本分类：IMDB（25,000训练，25,000测试，二分类）  
尝试了多个模型，效果最好的是最简单的[EmebddingBag](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)  
可能是任务比较简单，复杂的模型反而特别容易过拟合  
 | model | RNN | LSTM | Bidirectional-LSTM | Transformer | EmbeddingBag |     
 | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |  
 | accuracy | 74.57 | 83.26 | 84.32 | 80.42 | 87.35 | 

# 机器翻译：CMN-ENG（18,000训练，500验证，2636测试，英文到中文的翻译数据集）
预处理：给的数据集好像预处理部分有一些错误，所以我使用spacy重新进行分词并按8：1：1重新划分训练，测试和验证集  
使用了两个模型，分别是基于LSTM的seq2seq，和自己复现的transformer，精度分别为 **31.58%** 和 **51.55%**  
此外，用了spacy的词向量初始化embedding layer  
一些翻译效果展示：  
![image](https://github.com/cenlibin/deeplearning_assignments/blob/main/Images/translation_examples.png)  
绘制了一下attention的可视化（3个head）:   
![image](https://github.com/cenlibin/deeplearning_assignments/blob/main/Images/attention_heatmap.png)  
