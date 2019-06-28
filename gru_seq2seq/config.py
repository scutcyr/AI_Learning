#!/usr/bin/python3
# South China University of Technology
# Yirong Chen
# mail:eecyryou@mail.scut.edu.cn

import os
import torch
from preprocessing import Corpuspreprocessing

'''
修改CorpusDict可以增加个人的Corpus
'''
GCR_name = "GCR"
CorpusDict = {"Chatterbot":"chatterbot.tsv",
              "Douban":"douban_single_turn.tsv",
              "Ptt":"ptt.tsv",
              "Qingyun":"qingyun.tsv", 
              "Subtitle":"subtitle.tsv", 
              "Tieba":"tieba.tsv", 
              "Weibo":"weibo.tsv", 
              "Xiaohuangji":"xiaohuangji.tsv"}
'''
修改Filepath可以实现指定Corpus的路径
'''
Filepath = "./"  # 注意：路径最后需要带上/
'''
修改Corpus可以实现指定的Corpus作为对话系统的数据集，注意: Corpus需要取CorpusDict已经存在的键值
'''
Corpus = "Chatterbot"

'''
修改Corpus可以实现指定的Corpus作为对话系统的数据集，注意: Corpus需要取CorpusDict已经存在的键值
'''
Modelpath = "./"+Corpus+"/"

ModelpathDict = { "RNN":"./gcr_model/rnn_encoder_decoder/",
                  "LSTM":"./gcr_model/lstm_encoder_decoder/",
                  "GRU":"./gcr_model/gru_encoder_decoder/"}

def Check_Corpus(Filepath = Filepath, Corpus = Corpus):
	if os.path.exists(Filepath+CorpusDict[Corpus]) == False:
		print("Corpus输入非法!")
		print("没有"+Corpus+"这个语料库")
		print("请检查Filepath和Corpus参数!")
	else:
		print("Filepath:"+Filepath)
		print("Filepath检查完毕，正常！")
		print("Corpus:"+Corpus)
		print("Corpus检查完毕，正常！")

def Count_Linenum(Filepath):
    linenum = -1
    for linenum, line in enumerate(open(Filepath, 'rU')):
        pass
    linenum += 1
    return linenum

def Check_Preprocess(Filepath = Filepath, Corpus = Corpus):
	Check_Corpus(Filepath = Filepath, Corpus = Corpus)
	# 数据预处理
	if os.path.exists(Filepath+Corpus+"_"+"question"+'.vocab') == False:
		pre = Corpuspreprocessing(Filepath = Filepath, Corpus = Corpus, CorpusDict = CorpusDict)
		pre.generate()
		protype = "question"
		question_word_num = Count_Linenum(Filepath = Filepath+Corpus+"_"+protype+'.vocab')
		protype = "answer"
		answer_word_num = Count_Linenum(Filepath = Filepath+Corpus+"_"+protype+'.vocab')
	else:
		protype = "question"
		question_word_num = Count_Linenum(Filepath = Filepath+Corpus+"_"+protype+'.vocab')
		protype = "answer"
		answer_word_num = Count_Linenum(Filepath = Filepath+Corpus+"_"+protype+'.vocab')
	return question_word_num, answer_word_num


#Check_Corpus(Filepath = Filepath, Corpus = Corpus)
question_word_num, answer_word_num = Check_Preprocess(Filepath = Filepath, Corpus = Corpus)

