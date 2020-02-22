
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
    raw_sentence = re.findall("\w+[\、*[\“\w+\”]*\.*]*\w*",text)

    return raw_sentence


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
        self.model = gensim.models.Word2Vec.load(path)
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
            try:
                freq = self.model.wv.vocab[word].count/self.model.corpus_total_words #TODO: 如何处理“out of vocabulary”?
            except:
                freq = 1/self.model.corpus_total_words
            try:
                vw = self.model[word]
            except:
                vw = self.model['棼']
            vs = vs+(a/(a+freq))*vw
        if len(clean_sentence)==0:
            freq = 10000/self.model.corpus_total_words
            vw = self.model['棼']
            vs = (a/(a+freq))*vw
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



        
        
        
