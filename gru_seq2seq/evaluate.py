#!/usr/bin/python3
# South China University of Technology
# Yirong Chen
# mail:eecyryou@mail.scut.edu.cn

'''
# 用法简介：
依赖文件：config.py、preprocessing.py、seq2seq.py

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
import seq2seq.seq2seq as Encoder_Decoder

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
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device("cuda:"+re.split(r",",args.gpu_id)[0] if USE_CUDA else "cpu")
gpu_id = list(map(int, re.split(r",",args.gpu_id)))
print("当前GPU: ", torch.cuda.current_device())


if __name__ == '__main__':

    question_word_num, answer_word_num = config.Check_Preprocess(Filepath = args.Filepath, Corpus = args.Corpus)

    gcr = Encoder_Decoder(input_size = question_word_num,
                  hidden_size = args.hidden_size,
                  output_size = answer_word_num,
                  n_layers = args.n_layers,
                  dropout_p = args.dropout_p,
                  max_length = args.max_length,
                  max_epoches = args.max_epoches,
                  beam_search = args.beam_search,
                  rnn_type = args.rnn_type,
                  use_cuda = args.use_cuda,
                  model_path = "./model/"+args.Corpus+"/",
                  Corpus = args.Corpus,
                  Filepath = args.Filepath)
    
    gcr = torch.nn.DataParallel(gcr, device_ids = gpu_id)
    gcr.to(device)


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
    print("model_path:  ", "./model/"+args.Corpus+"/")
    print("Corpus:  ", args.Corpus)
    print("Filepath:  ", args.Filepath)

    if os.path.exists("./model/"+args.Corpus+"/") == False:
        os.mkdir("./model/"+args.Corpus+"/")


    netparam = open("./model/"+str(args.Corpus)+"/"+"Networkparameters.txt", "w")
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
    netparam.write("model_path:  "+str("./model/"+args.Corpus+"/")+"\n")
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
