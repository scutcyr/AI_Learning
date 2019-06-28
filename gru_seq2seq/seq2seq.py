#!/usr/bin/python3
# South China University of Technology
# Yirong Chen
# mail:eecyryou@mail.scut.edu.cn

'''
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

'''

import os
import re
import sys
import math
import time
import jieba
import torch
import config # 常规参数设置
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from preprocessing import Corpuspreprocessing

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # 方便定位报错信息
USE_CUDA = torch.cuda.is_available()

SOS_token = 2
EOS_token = 1

parser = argparse.ArgumentParser(description='manual to seq2seq.py')
parser.add_argument('--run_type', help="本文件运行模式，主要分为train和predict两种", type=str, default = "train")
parser.add_argument('--input_size', help="Encoder对应的词嵌入的词库大小，等于vocab的大小+1", type=int, default = config.question_word_num)
parser.add_argument('--hidden_size', help="隐层大小", type=int, default = 256)
parser.add_argument('--output_size', help="Decoder对应的词嵌入的词库大小，等于vocab的大小+1", type=int, default = config.answer_word_num)
parser.add_argument('--n_layers', help="Encoder/Decoder网络层数", type=int, default = 2)
parser.add_argument('--dropout_p', help="dropout概率", type=float, default = 0.25)
parser.add_argument('--max_length', help="最大长度", type=int, default = 32)
parser.add_argument('--max_epoches', help="最大epoches", type=int, default = 100000)
parser.add_argument('--beam_search', help="是否进行beam_search算法搜索", type=bool, default = True)
parser.add_argument('--use_cuda', help="是否使用CUDA训练", type=bool, default = USE_CUDA)
parser.add_argument('--model_path', help="训练好的模型路径，默认为: .model/+Corpus+/", type=str, default = config.Modelpath)
parser.add_argument('--Corpus', help="对话语料库名称，文件格式为.tsv，每一行为一个句子对，形式为:Q \t A, 可选择: Chatterbot、Douban、Ptt、Qingyun、Subtitle、Tieba、Weibo、Xiaohuangji", type=str, default = config.Corpus)
parser.add_argument('--Filepath', help="文件路径", type=str, default = config.Filepath)
parser.add_argument('--rnn_type', help="RNN结构，可以选择RNN、LSTM、GRU，默认为GRU", type=str, default = "GRU")
parser.add_argument('--gpu_id', help="GPU_ID", type=str, default = "0,1,2,3,4")
parser.add_argument('--evaluate', help="evaluate文件", type=str, default = "./evaluate.tsv")
parser.add_argument('--top_k', help="beam_search宽度", type=int, default = 5)
parser.add_argument('--alpha', help="惩罚因子", type=float, default = 0.5)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device("cuda:"+re.split(r",",args.gpu_id)[0] if USE_CUDA else "cpu")
gpu_id = list(map(int, re.split(r",",args.gpu_id)))
print("当前GPU: ", torch.cuda.current_device())

class EncoderRNN(nn.Module):
    def __init__(self, 
                 input_size = 1000, 
                 hidden_size = 128, 
                 encoder_layers = 2,
                 rnn_type = "GRU",
                 use_cuda = USE_CUDA):
        """EncoderRNN RNN编码器
        Args:
          input_size: Encoder对应的词嵌入的词库大小，等于vocab的大小+1
          hidden_size: 隐层大小
          encoder_layers：encoder中的RNN网络层数
          rnn_type: RNN结构，可以选择RNN、LSTM、GRU，默认为GRU
        """
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder_layers = encoder_layers
        self.use_cuda = use_cuda

        self.embedding = nn.Embedding(input_size, hidden_size)  # 词嵌入
        if rnn_type == "GRU":
            self.rnn = nn.GRU(hidden_size, hidden_size, encoder_layers)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(hidden_size, hidden_size, encoder_layers)
        elif rnn_type == "RNN":
            self.rnn = nn.RNN(hidden_size, hidden_size, encoder_layers)
        else:
            self.rnn = nn.RNN(hidden_size, hidden_size, encoder_layers)

    def forward(self, 
                word_inputs, 
                hidden):
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)  #Pytorch中 view()函数作用是将一个多行的Tensor,拼接成一行
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.encoder_layers, 1, self.hidden_size))
        if self.use_cuda: hidden = hidden.to(device)#.cuda()
        return hidden


class Attn(nn.Module):
    def __init__(self, 
                 method, 
                 hidden_size, 
                 max_length,
                 use_cuda = USE_CUDA):
        """Attn 注意力机
           Args:
            method: 实现注意力的方法
            hidden_size: 隐层大小
            max_length：最大长度
        """
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, 
                hidden, 
                encoder_outputs):
        seq_len = len(encoder_outputs)

        attn_energies = Variable(torch.zeros(seq_len)) # B x 1 x S
        if self.use_cuda: attn_energies = attn_energies.to(device)#.cuda()

        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        return F.softmax(attn_energies, dim = 0).unsqueeze(0).unsqueeze(0)

    def score(self, 
              hidden, 
              encoder_output):
        if self.method == 'dot':
            energy = torch.dot(hidden.view(-1), encoder_output.view(-1))
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = torch.dot(hidden.view(-1), encoder_output.view(-1))
            return energy

class AttnDecoderRNN(nn.Module):
    def __init__(self, 
                 attn_model, 
                 hidden_size, 
                 output_size, 
                 decoder_layers=1, 
                 dropout_p=0.1, 
                 max_length=10,
                 rnn_type = "GRU"):
        """AttnDecoderRNN Attention+RNN实现的Decoder
           Args:
            attn_model: Attention方法，可以选择'none'、'general'、'concat'、'dot'、'general'
            hidden_size: 隐层大小
            output_size：Decoder对应的词嵌入的词库大小，等于vocab的大小+1
            decoder_layers: dncoder中的RNN网络层数
            dropout_p: dropout率
            max_length: 最大长度
            rnn_type: RNN结构，可以选择RNN、LSTM、GRU，默认为GRU
        """
        super(AttnDecoderRNN, self).__init__()

        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.decoder_layers = decoder_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(output_size, hidden_size)

        if rnn_type == "GRU":
            self.rnn = nn.GRU(hidden_size * 2, hidden_size, decoder_layers, dropout=dropout_p)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(hidden_size * 2, hidden_size, decoder_layers, dropout=dropout_p)
        elif rnn_type == "RNN":
            self.rnn = nn.RNN(hidden_size * 2, hidden_size, decoder_layers, dropout=dropout_p)
        else:
            self.rnn = nn.RNN(hidden_size * 2, hidden_size, decoder_layers, dropout=dropout_p)

        self.out = nn.Linear(hidden_size * 2, output_size)

        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size, self.max_length)

    def forward(self, 
                word_input, 
                last_context, 
                last_hidden, 
                encoder_outputs):

        word_embedded = self.embedding(word_input).view(1, 1, -1) # S=1 x B x N  # 这部分可以使用BERT替代

        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        rnn_output, hidden = self.rnn(rnn_input, last_hidden)

        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N

        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)), dim = 1)
        #output = self.out(torch.cat((rnn_output, context), 1))
        return output, context, hidden, attn_weights


class seq2seq(nn.Module):
    def __init__(self,
                 input_size = config.question_word_num,
                 hidden_size = 256,
                 output_size = config.answer_word_num,
                 n_layers = 2,
                 dropout_p = 0.05,
                 max_length = 32,
                 max_epoches = 100000,
                 beam_search = True,
                 rnn_type = "GRU",
                 use_cuda = USE_CUDA,
                 model_path = config.Modelpath,
                 Corpus = config.Corpus,
                 Filepath = config.Filepath,
                 top_k = 5,
                 alpha = 0.5):
        """AttnDecoderRNN Attention+RNN实现的Decoder
           Args:
            input_size: Encoder对应的词嵌入的词库大小，等于vocab的大小+1
            hidden_size: 隐层大小
            output_size: Decoder对应的词嵌入的词库大小，等于vocab的大小+1
            n_layers: Encoder/Decoder网络层数
            dropout_p: dropout概率
            max_length：最大长度
            max_epoches: 最大epoches
            beam_search: 是否进行beam_search算法搜索
            rnn_type: RNN结构，可以选择RNN、LSTM、GRU，默认为GRU
            use_cuda: 是否使用CUDA训练
            model_path: 训练好的模型路径，默认为: ".model/"+Corpus+"/"
            Corpus: 对话语料库名称，文件格式为.tsv，每一行为一个句子对，形式为:Q \t A, 可选择: Chatterbot、Douban、Ptt、Qingyun、Subtitle、Tieba、Weibo、Xiaohuangji
            Filepath: 文件路径.

        """
        super(seq2seq, self).__init__()
        self.max_epoches = max_epoches
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.n_layers = n_layers
        self.rnn_type = rnn_type
        self.dropout_p = dropout_p
        self.beam_search = beam_search
        self.use_cuda = use_cuda
        self.model_path = model_path
        self.Filepath = Filepath
        self.Corpus = Corpus  # 可选择: Chatterbot、Douban、Ptt、Qingyun、Subtitle、Tieba、Weibo、Xiaohuangji
        
        self.show_epoch = 100
        self.batch_index = 0
        self.GO_token = 2
        self.EOS_token = 1
        self.top_k = top_k     # beam width(集束宽)
        self.alpha = alpha

        self.enc_vec = []
        self.dec_vec = []

        # 初始化encoder和decoder
        self.encoder = EncoderRNN(self.input_size, self.hidden_size, self.n_layers, rnn_type = self.rnn_type)
        self.decoder = AttnDecoderRNN('general', self.hidden_size, self.output_size, self.n_layers, self.dropout_p, self.max_length, rnn_type = self.rnn_type)

        if self.use_cuda:
            self.encoder = self.encoder.to(device)#.cuda()
            self.decoder = self.decoder.to(device)#.cuda()

        self.encoder_optimizer = optim.Adam(self.encoder.parameters())
        self.decoder_optimizer = optim.Adam(self.decoder.parameters())
        # 初始化损失函数
        self.criterion = nn.CrossEntropyLoss()  #nn.NLLLoss()

    def loadData(self, DoCorpuspreprocess = False):

        if DoCorpuspreprocess == True:
            pre = Corpuspreprocessing(Filepath = self.Filepath, Corpus = self.Corpus, CorpusDict = config.CorpusDict)
            pre.generate()

        with open(self.Filepath+self.Corpus+"_question"+".vec") as enc:
            line = enc.readline()
            while line:
                self.enc_vec.append(line.strip().split())
                line = enc.readline()

        with open(self.Filepath+self.Corpus+"_answer"+".vec") as dec:
            line = dec.readline()
            while line:
                self.dec_vec.append(line.strip().split())
                line = dec.readline()


    def next(self, batch_size, eos_token=1, go_token=2, shuffle=False):
        inputs = []
        targets = []

        if shuffle:
            ind = random.choice(range(len(self.enc_vec)))
            enc = [self.enc_vec[ind]]
            dec = [self.dec_vec[ind]]
        else:
            if self.batch_index+batch_size >= len(self.enc_vec):
                enc = self.enc_vec[self.batch_index:]
                dec = self.dec_vec[self.batch_index:]
                self.batch_index = 0
            else:
                enc = self.enc_vec[self.batch_index:self.batch_index+batch_size]
                dec = self.dec_vec[self.batch_index:self.batch_index+batch_size]
                self.batch_index += batch_size
        for index in range(len(enc)):
            enc = enc[0][:self.max_length] if len(enc[0]) > self.max_length else enc[0]
            dec = dec[0][:self.max_length] if len(dec[0]) > self.max_length else dec[0]

            enc = [int(i) for i in enc]
            dec = [int(i) for i in dec]
            dec.append(eos_token)

            inputs.append(enc)
            targets.append(dec)

        inputs = Variable(torch.LongTensor(inputs)).transpose(1, 0).contiguous()
        targets = Variable(torch.LongTensor(targets)).transpose(1, 0).contiguous()
        if self.use_cuda:
            inputs = inputs.to(device)#.cuda()
            targets = targets.to(device)#.cuda()
        return inputs, targets

    def train(self):
        self.loadData()
        try:
            self.load_state_dict(torch.load(self.model_path+'params.pkl'))
        except Exception as e:
            print(e)
            print("No model!")
        loss_track = []
        word_num_track = []
        total_loss = 0
        total_word = 0

        for epoch in range(self.max_epoches):
            start = time.time()
            inputs, targets = self.next(batch_size=1, shuffle=False)
            target_length = targets.size()[0]
            word_num_track.append(target_length) 
            loss, logits = self.step(inputs, targets, self.max_length)
            loss_track.append(loss)
            total_loss += target_length*loss
            total_word += target_length
            _,v = torch.topk(logits, 1)
            pre = v.cpu().data.numpy().T.tolist()[0][0]
            tar = targets.cpu().data.numpy().T.tolist()[0]
            stop = time.time()
            if epoch % self.show_epoch == 0:
                print("-"*50)
                print("epoch:", epoch)
                print("    loss:", loss)
                print("    ppl:", math.exp(min(loss, 100)))
                print("    total_loss:", total_loss/total_word)
                print("    total_ppl:", math.exp(min(total_loss/total_word, 100)))
                print("    target:%s\n    output:%s" % (tar, pre))
                print("    per-time:", (stop-start))
                torch.save(self.state_dict(), self.model_path+'params.pkl')

    def step(self, 
             input_variable, 
             target_variable, 
             max_length):
        teacher_forcing_ratio = 0.1
        clip = 5.0
        loss = 0 # Added onto for each word

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length = input_variable.size()[0]
        target_length = target_variable.size()[0]

        encoder_hidden = self.encoder.init_hidden()
        encoder_outputs, encoder_hidden = self.encoder(input_variable, encoder_hidden)

        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        decoder_context = Variable(torch.zeros(1, self.decoder.hidden_size))
        decoder_hidden = encoder_hidden # Use last hidden state from encoder to start decoder
        if self.use_cuda:
            decoder_input = decoder_input.to(device)#.cuda()
            decoder_context = decoder_context.to(device)#.cuda()

        decoder_outputs = []
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        use_teacher_forcing = True
        if use_teacher_forcing:
            for di in range(target_length):
                decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output, target_variable[di])
                decoder_input = target_variable[di]
                decoder_outputs.append(decoder_output.unsqueeze(0))
        else:
            for di in range(target_length):
                decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output, target_variable[di])
                decoder_outputs.append(decoder_output.unsqueeze(0))
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]

                decoder_input = Variable(torch.LongTensor([[ni]]))
                if self.use_cuda: decoder_input = decoder_input.to(device)#.cuda()
                if ni == EOS_token: break
        loss.backward()
        #torch.nn.utils.clip_grad_norm(self.encoder.parameters(), clip)
        #torch.nn.utils.clip_grad_norm(self.decoder.parameters(), clip)
        # for pytorch >=1.0.0
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        decoder_outputs = torch.cat(decoder_outputs, 0)
        return loss.item() / target_length, decoder_outputs   # loss.data[0]改为loss.item() 以解决这个错误： #IndexError: invalid index of a 0-dim tensor. Use tensor.item() to convert a 0-dim tensor to a Python number

    def make_infer_fd(self, input_vec):
        inputs = []
        enc = input_vec[:self.max_length] if len(input_vec) > self.max_length else input_vec
        inputs.append(enc)
        inputs = Variable(torch.LongTensor(inputs)).transpose(1, 0).contiguous()
        if self.use_cuda:
            inputs = inputs.to(device)#.cuda()
        return inputs

    def predict(self):
        try:
            self.load_state_dict(torch.load(self.model_path+'params.pkl'))
        except Exception as e:
            print(e)
            print("No model!")
        '''
        for state in self.state.values():
            for k, v in state.items():
                print (type(v))
                if torch.is_tensor(v):
                   state[k] = v.cuda(cuda_id)
        '''
        loss_track = []

        # 加载字典
        str_to_vec = {}
        with open(self.Filepath+self.Corpus+"_question"+".vocab") as enc_vocab:
            for index,word in enumerate(enc_vocab.readlines()):
                str_to_vec[word.strip()] = index

        vec_to_str = {}
        with open(self.Filepath+self.Corpus+"_answer"+".vocab") as dec_vocab:
            for index,word in enumerate(dec_vocab.readlines()):
                vec_to_str[index] = word.strip()

        while True:
            input_strs = input("me"+" > ")
            # 字符串转向量
            segement = jieba.lcut(input_strs)
            input_vec = [str_to_vec.get(i, 3) for i in segement]
            input_vec = self.make_infer_fd(input_vec)

            # inference
            if self.beam_search:
                samples = self.beamSearchDecoder(input_vec)
                for sample in samples:
                    outstrs = []
                    for i in sample[0]:
                        if i == 1:
                            break
                        outstrs.append(vec_to_str.get(i, "Un"))
                    print(config.GCR_name+" > ", "".join(outstrs), sample[3])
            else:
                logits = self.infer(input_vec)
                _,v = torch.topk(logits, 1)
                pre = v.cpu().data.numpy().T.tolist()[0][0]
                outstrs = []
                for i in pre:
                    if i == 1:
                        break
                    outstrs.append(vec_to_str.get(i, "Un"))
                print(config.GCR_name+" > ", "".join(outstrs))

    def infer(self, input_variable):
        input_length = input_variable.size()[0]

        encoder_hidden = self.encoder.init_hidden()
        encoder_outputs, encoder_hidden = self.encoder(input_variable, encoder_hidden)

        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        decoder_context = Variable(torch.zeros(1, self.decoder.hidden_size))
        decoder_hidden = encoder_hidden
        if self.use_cuda:
            decoder_input = decoder_input.to(device)#.cuda()
            decoder_context = decoder_context.to(device)#.cuda()
        decoder_outputs = []

        for i in range(self.max_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            decoder_outputs.append(decoder_output.unsqueeze(0))
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            decoder_input = Variable(torch.LongTensor([[ni]])) # Chosen word is next input
            if self.use_cuda: decoder_input = decoder_input.to(device)#.cuda()
            if ni == EOS_token: break

        decoder_outputs = torch.cat(decoder_outputs, 0)
        return decoder_outputs

    def tensorToList(self, tensor):
        return tensor.cpu().data.numpy().tolist()[0]

    def beamSearchDecoder(self, input_variable):
        input_length = input_variable.size()[0]
        encoder_hidden = self.encoder.init_hidden()
        encoder_outputs, encoder_hidden = self.encoder(input_variable, encoder_hidden)

        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        decoder_context = Variable(torch.zeros(1, self.decoder.hidden_size))
        decoder_hidden = encoder_hidden
        if self.use_cuda:
            decoder_input = decoder_input.to(device)#.cuda()
            decoder_context = decoder_context.to(device)#.cuda()

        decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
        topk = decoder_output.data.topk(self.top_k)
        samples = [[] for i in range(self.top_k)]
        dead_k = 0
        final_samples = []
        for index in range(self.top_k):
            topk_prob = topk[0][0][index]
            topk_index = int(topk[1][0][index])
            samples[index] = [[topk_index], topk_prob, 0, 0, decoder_context, decoder_hidden, decoder_attention, encoder_outputs]

        for _ in range(self.max_length):
            tmp = []
            for index in range(len(samples)):
                tmp.extend(self.beamSearchInfer(samples[index], index))
            samples = []

            # 筛选出topk
            df = pd.DataFrame(tmp)
            df.columns = ['sequence', 'pre_socres', 'fin_scores', "ave_scores", "decoder_context", "decoder_hidden", "decoder_attention", "encoder_outputs"]
            sequence_len = df.sequence.apply(lambda x:len(x))
            df['ave_scores'] = df['fin_scores'] / sequence_len
            df = df.sort_values('ave_scores', ascending=False).reset_index().drop(['index'], axis=1)
            df = df[:(self.top_k-dead_k)]
            for index in range(len(df)):
                group = df.ix[index]
                if group.tolist()[0][-1] == 1:
                    final_samples.append(group.tolist())
                    df = df.drop([index], axis=0)
                    dead_k += 1
                    print("drop {}, {}".format(group.tolist()[0], dead_k))
            samples = df.values.tolist()
            if len(samples) == 0:
                break

        if len(final_samples) < self.top_k:
            final_samples.extend(samples[:(self.top_k-dead_k)])
        return final_samples

    def beamSearchInfer(self, sample, k):
        samples = []
        decoder_input = Variable(torch.LongTensor([[sample[0][-1]]]))
        if self.use_cuda:
            decoder_input = decoder_input.to(device)#.cuda()
        sequence, pre_scores, fin_scores, ave_scores, decoder_context, decoder_hidden, decoder_attention, encoder_outputs = sample
        decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)

        # choose topk
        topk = decoder_output.data.topk(self.top_k)
        for k in range(self.top_k):
            topk_prob = topk[0][0][k]
            topk_index = int(topk[1][0][k])
            pre_scores += topk_prob
            fin_scores = pre_scores - (k - 1 ) * self.alpha
            samples.append([sequence+[topk_index], pre_scores, fin_scores, ave_scores, decoder_context, decoder_hidden, decoder_attention, encoder_outputs])
        return samples

    def retrain(self):
        try:
            os.remove(self.model_path)
        except Exception as e:
            pass
        self.train()


    def evaluate_next(self, input_vec_list, output_vec_list, batch_size, eos_token=1, go_token=2, shuffle=False):
        inputs = []
        targets = []

        if shuffle:
            ind = random.choice(range(len(input_vec_list)))
            enc = [input_vec_list[ind]]
            dec = [output_vec_list[ind]]
        else:
            if self.batch_index+batch_size >= len(output_vec_list):
                enc = input_vec_list[self.batch_index:]
                dec = output_vec_list[self.batch_index:]
                self.batch_index = 0
            else:
                enc = input_vec_list[self.batch_index:self.batch_index+batch_size]
                dec = output_vec_list[self.batch_index:self.batch_index+batch_size]
                self.batch_index += batch_size
        for index in range(len(enc)):
            enc = enc[0][:self.max_length] if len(enc[0]) > self.max_length else enc[0]
            dec = dec[0][:self.max_length] if len(dec[0]) > self.max_length else dec[0]

            enc = [int(i) for i in enc]
            dec = [int(i) for i in dec]
            dec.append(eos_token)

            inputs.append(enc)
            targets.append(dec)

        inputs = Variable(torch.LongTensor(inputs)).transpose(1, 0).contiguous()
        targets = Variable(torch.LongTensor(targets)).transpose(1, 0).contiguous()
        if self.use_cuda:
            inputs = inputs.to(device)#.cuda()
            targets = targets.to(device)#.cuda()
        return inputs, targets



    def evaluate_step(self, 
             input_variable, 
             target_variable, 
             max_length):
        teacher_forcing_ratio = 0.1
        clip = 5.0
        loss = 0 # Added onto for each word

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length = input_variable.size()[0]
        target_length = target_variable.size()[0]

        encoder_hidden = self.encoder.init_hidden()
        encoder_outputs, encoder_hidden = self.encoder(input_variable, encoder_hidden)

        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        decoder_context = Variable(torch.zeros(1, self.decoder.hidden_size))
        decoder_hidden = encoder_hidden # Use last hidden state from encoder to start decoder
        if self.use_cuda:
            decoder_input = decoder_input.to(device)#.cuda()
            decoder_context = decoder_context.to(device)#.cuda()

        decoder_outputs = []
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        use_teacher_forcing = True
        if use_teacher_forcing:
            for di in range(target_length):
                decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output, target_variable[di])
                decoder_input = target_variable[di]
                decoder_outputs.append(decoder_output.unsqueeze(0))
        else:
            for di in range(target_length):
                decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output, target_variable[di])
                decoder_outputs.append(decoder_output.unsqueeze(0))
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]

                decoder_input = Variable(torch.LongTensor([[ni]]))
                if self.use_cuda: decoder_input = decoder_input.to(device)#.cuda()
                if ni == EOS_token: break
        #loss.backward()
        #torch.nn.utils.clip_grad_norm(self.encoder.parameters(), clip)
        #torch.nn.utils.clip_grad_norm(self.decoder.parameters(), clip)
        # for pytorch >=1.0.0
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        decoder_outputs = torch.cat(decoder_outputs, 0)
        return loss.item() / target_length, decoder_outputs   # loss.data[0]改为loss.item() 以解决这个错误： #IndexError: invalid index of a 0-dim tensor. Use tensor.item() to convert a 0-dim tensor to a Python number


    def evaluate(self, evaluate_file):
        try:
            self.load_state_dict(torch.load(self.model_path+'params.pkl'))
        except Exception as e:
            print(e)
            print("No model!")

        loss = 0
        ppl = 0

        # 加载字典
        str_to_vec = {}
        with open(self.Filepath+self.Corpus+"_question"+".vocab") as enc_vocab:
            for index,word in enumerate(enc_vocab.readlines()):
                str_to_vec[word.strip()] = index

        vec_to_str = {}
        with open(self.Filepath+self.Corpus+"_answer"+".vocab") as dec_vocab:
            for index,word in enumerate(dec_vocab.readlines()):
                vec_to_str[index] = word.strip()
        input_vec_list = []
        output_vec_list = []
        with open(evaluate_file, 'r') as evaluate_corpus:
            for sent in evaluate_corpus.readlines():
                sent_q = re.split(r'\t',sent)[0]
                sent_a = re.split(r'\t',sent)[1]
                sent_q = jieba.lcut(sent_q)
                input_vec = [str_to_vec.get(i, 3) for i in sent_q]
                input_vec_list.append(input_vec)
                sent_a = jieba.lcut(sent_a)
                output_vec = [str_to_vec.get(i, 3) for i in sent_a]
                output_vec_list.append(output_vec)

        loss_track = []
        word_num_track = []
        total_loss = 0
        total_word = 0

        for epoch in range(self.max_epoches):
            start = time.time()
            inputs, targets = self.evaluate_next(input_vec_list=input_vec_list, output_vec_list=output_vec_list, batch_size=1, shuffle=False)
            target_length = targets.size()[0]
            word_num_track.append(target_length) 
            loss, logits = self.evaluate_step(inputs, targets, self.max_length)
            loss_track.append(loss)
            total_loss += target_length*loss
            total_word += target_length
            _,v = torch.topk(logits, 1)
            pre = v.cpu().data.numpy().T.tolist()[0][0]
            tar = targets.cpu().data.numpy().T.tolist()[0]
            stop = time.time()
            if epoch % self.show_epoch == 0:
                print("-"*50)
                print("epoch:", epoch)
                print("    loss:", loss)
                print("    ppl:", math.exp(min(loss, 100)))
                print("    total_loss:", total_loss/total_word)
                print("    total_ppl:", math.exp(min(total_loss/total_word, 100)))
                print("    target:%s\n    output:%s" % (tar, pre))
                print("    per-time:", (stop-start))
                torch.save(self.state_dict(), self.model_path+'params.pkl')
        total_ppl = math.exp(min(total_loss/total_word, 100))
        return loss_track,total_loss,total_ppl,total_word



if __name__ == '__main__':

    question_word_num, answer_word_num = config.Check_Preprocess(Filepath = args.Filepath, Corpus = args.Corpus)

    seq = seq2seq(input_size = question_word_num,
                  hidden_size = args.hidden_size,
                  output_size = answer_word_num,
                  n_layers = args.n_layers,
                  dropout_p = args.dropout_p,
                  max_length = args.max_length,
                  max_epoches = args.max_epoches,
                  beam_search = args.beam_search,
                  rnn_type = args.rnn_type,
                  use_cuda = args.use_cuda,
                  model_path = config.ModelpathDict[args.rnn_type]+args.Corpus+"/",
                  Corpus = args.Corpus,
                  Filepath = args.Filepath,
                  top_k = args.top_k,
                  alpha = args.alpha)
    
    seq = torch.nn.DataParallel(seq, device_ids = gpu_id)
    seq.to(device)


    print("网络参数如下:  ")
    print("input_size:  ", question_word_num)
    print("hidden_size:  ", args.hidden_size)
    print("output_size:  ", answer_word_num)
    print("n_layers:  ", args.n_layers)
    print("dropout_p:  ", args.dropout_p)
    print("max_length:  ", args.max_length)
    print("max_epoches:  ", args.max_epoches)
    print("beam_search:  ", args.beam_search)
    print("rnn_type:  ", args.rnn_type)
    print("use_cuda:  ", args.use_cuda)
    print("model_path:  ", config.ModelpathDict[args.rnn_type]+args.Corpus+"/")
    print("Corpus:  ", args.Corpus)
    print("Filepath:  ", args.Filepath)

    if os.path.exists(config.ModelpathDict[args.rnn_type]+args.Corpus+"/") == False:
        os.mkdir(config.ModelpathDict[args.rnn_type]+args.Corpus+"/")


    netparam = open(config.ModelpathDict[args.rnn_type]+args.Corpus+"/"+"Networkparameters.txt", "w")
    netparam.write("input_size:  "+str(question_word_num)+"\n")
    netparam.write("hidden_size:  "+str(args.hidden_size)+"\n")
    netparam.write("output_size:  "+str(answer_word_num)+"\n")
    netparam.write("n_layers:  "+str(args.n_layers)+"\n")
    netparam.write("dropout_p:  "+str(args.dropout_p)+"\n")
    netparam.write("max_length:  "+str(args.max_length)+"\n")
    netparam.write("imax_epoches:  "+str(args.max_epoches)+"\n")
    netparam.write("beam_search:  "+str(args.beam_search)+"\n")
    netparam.write("rnn_type:  "+str(args.rnn_type)+"\n")
    netparam.write("use_cuda:  "+str(args.use_cuda)+"\n")
    netparam.write("model_path:  "+str(config.ModelpathDict[args.rnn_type]+args.Corpus+"/")+"\n")
    netparam.write("Corpus:  "+str(args.Corpus)+"\n")
    netparam.write("Filepath:  "+str(args.Filepath)+"\n")
    netparam.close()

    if args.run_type == 'train':
        #seq.train()  # 单GPU
        seq.module.train()   # 加上.module
    elif args.run_type == 'predict':
        #seq.predict()  # 单GPU
        seq.module.predict()   # 加上.module
    elif args.run_type == 'retrain':
        #seq.retrain()  # 单GPU
        seq.module.retrain()   # 加上.module
    elif args.run_type == 'evaluate':
        seq.module.evaluate(args.evaluate) 

