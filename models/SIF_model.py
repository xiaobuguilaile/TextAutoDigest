
'''
@File       : SIF_model.py
@Author     : Zeen Song
@Date       : 2020/2/21
@Desc       : Build Sentence Embedding using SIF methods
'''

import numpy as np
import jieba
import sklearn
import gensim
import pandas as pd
import os
import re
# cwd = os.getcwd()
# jieba.load_userdict(cwd+"/utils/jieba_latest_dict.txt")


def cleanReviewChinese(content):
    """
    中文文本预处理函数：去除各种标点符号，HTML标签，小写化
    :param content: str
    :return: List[str]
    """

    # 去除标点,数字, 字母及特殊字符
    sub_text = re.sub('[a-zA-Z0-9'
                      '１２３４５６７８９０'
                      'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ'
                      'ｅｓｎｇｒｉｋｄｃｈａｔｖｘｙｚｏｌｍｐｆｕ'
                      '!；：．·、…【】《》“”‘’！？"#$%&％\'?@，。〔〕［］（）()*+,-——\\./:;<=>＋×／'
                      '①↑↓★▌▲●℃[\\]^_`{|}~\s]+',
                      "",
                      content)

    # 去除不可见字符
    newContent = re.sub('[\001\002\003\004\005\006\007\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a]+', '', sub_text)

    # 以空格划分单词
    words = jieba.cut(newContent)

    # 去除停止词
    with open("./data/stopwords.txt", encoding="utf-8-sig") as f:
        stopwords = [line.strip() for line in f.readlines()]
    filter_words = [word for word in words if word not in stopwords]

    return filter_words


def text2sentence(text):
    '''
    将文本切分为句子，任意标点分割 (待定)
    -------------------------------
    text: str
    :rtype: List[str]
    '''
    text = re.sub('\n','',text)
    text = re.sub('\[\w+\]\:*\w*','',text)
    text = re.sub('([。！？\?])([^”’])', r"\1\n\2", text)  # 单字符断句符
    text = re.sub('(\.{6})([^”’])', r"\1\n\2", text)  # 英文省略号
    text = re.sub('(\…{2})([^”’])', r"\1\n\2", text)  # 中文省略号
    text = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', text)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    text = text.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return text.split("\n")

def Get_Score(Vs,Vt,Vc,w1=0.4,w2=0.6):
    '''
    通过计算每一个句子向量与标题、全文的加权余弦近似度，通过sigmoid函数将结果映射到0~1之间
    -------------------------------------------------------------------------
    Vs: Dict{str:np.array}
    Vt,Vc: np.array (分别为标题，全文向量)
    w1: float (标题的加权)
    w2: float (全文的加权)
    ：rtype: Dict{str:float}
    '''
    Score_dict = {}
    for (sentence,vs) in Vs.items():
        Score_dict[sentence] = w1*sigmoid(vs.dot(Vt))+w2*sigmoid(vs.dot(Vc))
    return Score_dict

def sigmoid(x):
    '''
    计算sigmoid函数
    '''
    return 1/(1+np.exp(-x))

def do_KNN(Score_dict,sentences,w_C=2,w_O=5,k=1):
    '''
    对每一个分数取前后一句话的加权
    --------------------------
    Score_dict: Dict{str:float}
    sentences: List[str]
    w_C: float (上下文(context)的权重)
    w_O: float (中心句(observation)的权重)
    k: int (k个近邻)
    '''
    new_dict = {}
    for i,sentence in enumerate(sentences):
        if i<k:
            new_dict[sentence] = w_O*Score_dict[sentence]
            for j in range(1,k+1):
                new_dict[sentence] += w_C*Score_dict[sentences[i+j]]
            new_dict[sentence] = new_dict[sentence]/(w_O+k*w_C)
        elif i>=len(sentences)-k:
            new_dict[sentence] = w_O*Score_dict[sentence]
            for j in range(1,k+1):
                new_dict[sentence] += w_C*Score_dict[sentences[i-j]]
            new_dict[sentence] = new_dict[sentence]/(w_O+k*w_C)
        else:
            new_dict[sentence] = w_O*Score_dict[sentence]
            for j in range(1,k+1):
                new_dict[sentence] += w_C*Score_dict[sentences[i-j]]+w_C*Score_dict[sentences[i+j]]
            new_dict[sentence] = new_dict[sentence]/(w_O+2*k*w_C)
    return new_dict

def summarize(Score_dict,sentences,k=20):
    '''
    根据打分输出句子
    -----------------
    Score_dict: Dict{str:float}
    sentences: List[str]
    k: int (需要输出的句子数量)
    '''
    sorted_sent = sorted(Score_dict.items(),key=lambda x: x[1],reverse=True)
    lower_bound = Score_dict[sorted_sent[k][0]]
    output = []
    for sentence in sentences:
        if Score_dict[sentence] > lower_bound:
            output.append(sentence)
    return output
    
class GETSentence_Embedding():
    '''
    通过输入的句子产生句向量
    '''
    def __init__(self,path):
        '''
        初始化
        --------
        path: str (词向量的路径)
        '''
        self.model = gensim.models.KeyedVectors.load(path)
        self.sentence = ''
        self.singular_vector = np.empty((self.model.wv.vector_size,self.model.wv.vector_size))

    def _process_sentence(self):
        '''
        使用与预处理相同的方法进行分词
        -------------------------
        :rtype: List[str]
        '''
        return cleanReviewChinese(self.sentence)

    def load_data(self,sentence):
        '''
        读取句子
        --------
        sentence: str (目标句子)
        '''

        self.sentence = sentence

    def sentence_EMB(self,a=0.0001):
        '''
        使用SIF方法获得加权句向量 注: 此处未进行SVD
        -------------------------------------
        a: float (可调参数)
        :rtype: np.array
        '''
        clean_sentence = self._process_sentence()
        vs = np.zeros(self.model.wv.vector_size)
        for word in clean_sentence:
            # use laplace smoothing for OOV word
            try:
                freq = (1+self.model.wv.vocab[word].count)/(self.model.corpus_total_words+self.model.corpus_count) 
            except:
                freq = 1/(self.model.corpus_total_words+self.model.corpus_count)
            try:
                vw = self.model[word]
            except:
                vw = np.random.rand(self.model.wv.vector_size)

            vs = vs+(a/(a+freq))*vw

        if len(clean_sentence)==0: #if the sentence is empty after stopwords-removal
            vs = np.random.rand(self.model.wv.vector_size)
            return vs

        vs = vs/len(clean_sentence)

        return vs

    def do_SVD(self,VS):
        '''
        对整个文章的句向量进行SVD，每一个句向量减去第一个特征向量
        ------------------------------------------------
        VS: np.array
        '''
        [U,_,_] = np.linalg.svd(VS)
        self.singular_vector = U






        
        
        
