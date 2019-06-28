#!/usr/bin/python3
# South China University of Technology
# Yirong Chen
# mail:eecyryou@mail.scut.edu.cn
import re
import jieba
#import config # 常用参数配置文件

class Corpuspreprocessing():
    __PAD__ = 0
    __GO__ = 1
    __EOS__ = 2
    __UNK__ = 3
    vocab = ['__PAD__', '__GO__', '__EOS__', '__UNK__']
    def __init__(self,
                 Filepath = None,
                 Corpus = None,
                 CorpusDict = None):
        """preprocessing 数据预处理类
        Args:
          Filepath: 文件路径.
          Corpus: 对话语料库名称，文件格式为.tsv，每一行为一个句子对，形式为:Q \t A
          例如：
            你是谁?    谁? 谁只是代表了一个人罢了
            你是谁?    那么你呢?
            你是谁?    一个个戴面具的男人.
            你是谁?    我看得出来.
        """
        self.Corpus = Corpus
        self.CorpusFile = Filepath+CorpusDict[Corpus]
        self.savePath = Filepath
    
    def wordToVocabulary(self, 
                         originFile, 
                         vocabFile, 
                         segementFile, 
                         processtype):
        """wordToVocabulary函数
        Args:
          originFile: 原始语料文件路径，注意输入的文件的每一行为一个句子对，形式为:Q \t A
          vocabFile: 词汇表文件
          segementFile: 经过分词处理后的句子
          processtype: 处理类型，由于 originFile的每一行为一个句子对，形式为:Q \t A，我们需要拆分问题和答案，所以这里由"question"和"answer"决定
        """
        vocabulary = []
        sege = open(segementFile, "w")
        with open(originFile, 'r') as corpus:
            if processtype == "question":
                for sent in corpus.readlines():
                    sent = re.split(r'\t',sent)[0]
                    # 去标点
                    sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。“”’‘？?、~@#￥%……&*（）]+", "", sent.strip())
                    #sentence = sent.strip()
                    words = jieba.lcut(sentence)
                    vocabulary.extend(words)
                    for word in words:
                        sege.write(word+" ")
                    sege.write("\n")
            elif processtype == "answer":
                for sent in corpus.readlines():
                    sent = re.split(r'\t',sent)[1]
                    # 去标点
                    sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。“”’‘？?、~@#￥%……&*（）]+", "", sent.strip())
                    #sentence = sent.strip()
                    words = jieba.lcut(sentence)
                    vocabulary.extend(words)
                    for word in words:
                        sege.write(word+" ")
                    sege.write("\n")
            else:
                print("processtype的输入参数非法，请检查processtype是否为question或answer当中的一个!")
                raise
        sege.close()

        # 去重并存入词典
        vocab_file = open(vocabFile, "w")
        _vocabulary = list(set(vocabulary))
        _vocabulary.sort(key=vocabulary.index)
        _vocabulary = self.vocab + _vocabulary
        for index, word in enumerate(_vocabulary):
            vocab_file.write(word+"\n")
        vocab_file.close()

    def WordToVec(self, 
                  segementFile, 
                  vocabFile, 
                  VecFile):
        """WordtoVec函数
        Args:
          vocabFile: 词汇表文件
          segementFile: 经过分词处理后的句子
          VecFile: 向量化的句子文件
        """
        word_dicts = {}
        vec = []
        with open(vocabFile, "r") as dict_f:
            for index, word in enumerate(dict_f.readlines()):
                word_dicts[word.strip()] = index

        f = open(VecFile, "w")
        if "question.vec" in VecFile:
            f.write("3 3 3 3\n")
            f.write("3\n")
        elif "answer.vec" in VecFile:
            f.write(str(word_dicts.get("other", 3))+"\n")
            f.write(str(word_dicts.get("other", 3))+"\n")
        with open(segementFile, "r") as sege_f:
            for sent in sege_f.readlines():
                sents = [i.strip() for i in sent.split(" ")[:-1]]
                vec.extend(sents)
                for word in sents:
                    f.write(str(word_dicts.get(word))+" ")
                f.write("\n")
        f.close()

    def FileClean(self, 
                  originFile,
                  cleanedFile,
                  min_sentence_len = 3,
                  max_sentence_len = 256):
        """FileClean函数: 对文件originFile的每一行进行检索，并且删除每一行当中Q的字符数小于10或者A的字符数小于10的数据
        Args:
          originFile: 原始语料文件路径，注意输入的文件的每一行为一个句子对，形式为:Q \t A
          cleanedFile: 经过清洗后的数据
        """
        sege = open(cleanedFile, "w")
        with open(originFile, 'r') as corpus:
                for sent in corpus.readlines():
                    sent_q = re.split(r'\t',sent)[0]
                    sent_a = re.split(r'\t',sent)[1]
                    # 去标点
                    sentence_q = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。“”’‘？?、~@#￥%……&*（）]+", "", sent_q.strip())
                    sentence_a = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。“”’‘？?、~@#￥%……&*（）]+", "", sent_a.strip())
                    if len(sentence_q)>=min_sentence_len and len(sentence_a)>=min_sentence_len and len(sentence_q)<=max_sentence_len and len(sentence_a)<=max_sentence_len:
                      sege.write(sent_q+"\t"+sent_a)
        sege.close()
            
    def generate(self):

        # 清洗数据
        self.FileClean(originFile = self.CorpusFile,
                       cleanedFile = self.savePath+self.Corpus+"_"+"cleaned"+'.tsv',
                       min_sentence_len = 3,
                       max_sentence_len = 256)
        # 获得字典
        protype = "question"

        self.wordToVocabulary(originFile = self.savePath+self.Corpus+"_"+"cleaned"+'.tsv', 
                              vocabFile = self.savePath+self.Corpus+"_"+protype+'.vocab', 
                              segementFile = self.savePath+self.Corpus+"_"+protype+'.segement', 
                              processtype = protype )
        # 转向量
        self.WordToVec(segementFile = self.savePath+self.Corpus+"_"+protype+'.segement', 
                       vocabFile = self.savePath+self.Corpus+"_"+protype+'.vocab', 
                       VecFile = self.savePath+self.Corpus+"_"+protype+".vec")
        # 获得字典
        protype = "answer"
        self.wordToVocabulary(originFile = self.savePath+self.Corpus+"_"+"cleaned"+'.tsv', 
                              vocabFile = self.savePath+self.Corpus+"_"+protype+'.vocab', 
                              segementFile = self.savePath+self.Corpus+"_"+protype+'.segement', 
                              processtype = protype )
        # 转向量
        self.WordToVec(segementFile = self.savePath+self.Corpus+"_"+protype+'.segement', 
                       vocabFile = self.savePath+self.Corpus+"_"+protype+'.vocab', 
                       VecFile = self.savePath+self.Corpus+"_"+protype+".vec")

if __name__ == '__main__':
  # Corpuspreprocessing测试
  '''
  修改CorpusDict可以增加个人的Corpus
  '''
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
  Filepath = "./"
  '''
  修改Corpus可以实现指定的Corpus作为对话系统的数据集，注意: Corpus需要取CorpusDict已经存在的键值
  '''
  Corpus = "Weibo"
  pre = Corpuspreprocessing(Filepath = Filepath, Corpus = Corpus, CorpusDict = CorpusDict)
  pre.generate()
