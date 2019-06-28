# [GCR](https://github.com/scutcyr/GCR)
Author: [Yirong Chen](https://scutcyr.github.io/)
This is the AI project of Yirong Chen.
## Group Chat Robot    


# 用法简介：
依赖文件：config.py、preprocessing.py
## 在命令行窗口运行模板
python seq2seq.py --args_name parameter
## 例子1：使用默认的对话语料
训练对话系统网络
python seq2seq.py
测试对话系统网络
python seq2seq.py --run_type=predict
## 例子2：指定语料库路径和语料库名称
步骤一：打开config.py文件，在CorpusDict当中添加你的语料库，格式参考如下："Chatterbot":"chatterbot.tsv"
步骤二：
训练对话系统网络:
python seq2seq.py --Corpus=[CorpusName] --Filepath=[Filepath]
测试对话系统网络:
python seq2seq.py --Corpus=[CorpusName] --Filepath=[Filepath] --run_type=predict

例如，我的Corpus保存路径为：/157Dataset/data-chen.yirong/nlpdataset/dialogue/chat_corpus/chinese_corpus/clean_chat_corpus/
用于训练的语料集在config.py中的CorpusDict定义为"Chatterbot":"chatterbot.tsv"
则可以通过下列命令实现训练和测试
训练：
python seq2seq.py --Corpus=Chatterbot --Filepath=/157Dataset/data-chen.yirong/nlpdataset/dialogue/chat_corpus/chinese_corpus/clean_chat_corpus/
测试：
python seq2seq.py --Corpus=Chatterbot --Filepath=/157Dataset/data-chen.yirong/nlpdataset/dialogue/chat_corpus/chinese_corpus/clean_chat_corpus/ --run_type=predict

## 还可以使用--input_size、--hidden_size、--output_size、--n_layers、--dropout_p、--max_length、--max_epoches、--beam_search、--use_cuda、--rnn_type对网络参数进行设置

--run_type: 本文件运行模式，主要分为train和predict两种
--input_size: Encoder对应的词嵌入的词库大小，等于vocab的大小+1
--hidden_size: 隐层大小
--output_size: Decoder对应的词嵌入的词库大小，等于vocab的大小+1
--n_layers: Encoder/Decoder网络层数
--dropout_p: dropout概率
--max_length：最大长度
--max_epoches: 最大epoches
--beam_search: 是否进行beam_search算法搜索
--rnn_type: RNN结构，可以选择RNN、LSTM、GRU，默认为GRU
--use_cuda: 是否使用CUDA训练
--model_path: 训练好的模型路径，默认为: ".model/"+Corpus+"/"
--Corpus: 对话语料库名称，文件格式为.tsv，每一行为一个句子对，形式为:Q \t A, 可选择: Chatterbot、Douban、Ptt、Qingyun、Subtitle、Tieba、Weibo、Xiaohuangji
--Filepath: 文件路径.
