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


def cleanReviewChinese(content, stopwords_path="./data/stopwords.txt"):
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
    with open(stopwords_path, encoding="utf-8-sig") as f:
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
    对每一个分数取前后k句话的加权
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
    def __init__(self, path, score_title_weight=0.4, knn_w_c=2, knn_w_o=5, knn_k=2, sentence_embed_a=1e-4,
                 abstract_percent=0.2, max_output_length=20, stopwords_path="./data/stopwords.txt"):
        '''
        初始化
        --------
        path: str (词向量的路径)
        score_title_weight: float (最终得分计算中标题的加权)
        knn_w_c: float (knn计算，上下文(context)的权重)
        knn_w_o: float (中心句(observation)的权重)
        knn_k: int (knn计算，近邻个数)
        sentence_embed_a: float (使用SIF方法获得加权句向量的权重参数)
        abstract_percent: float (默认获取摘要占全文的百分比)
        max_output_length: int (最大摘要句子数目)
        stopwords_path: str (停用词表路径)
        '''
        self.model = gensim.models.Word2Vec.load(path)
        self.word_vector_size = self.model.wv.vector_size
        self.sentence = ''
        self.singular_vector = np.empty((self.word_vector_size,self.word_vector_size))
        self.score_title_weight = score_title_weight
        self.knn_w_c = knn_w_c
        self.knn_w_o = knn_w_o
        self.knn_k = knn_k
        self.sentence_embed_a = sentence_embed_a
        self.abstract_percent = abstract_percent
        self.max_output_length = max_output_length
        self.stopwords_path = stopwords_path

    def _preprocess_sentence(self):
        '''
        使用与预处理相同的方法进行分词
        -------------------------
        :rtype: List[str]
        '''
        return cleanReviewChinese(self.sentence, self.stopwords_path)

    def load_data(self,sentence):
        '''
        读取句子
        --------
        sentence: str (目标句子)
        '''

        self.sentence = sentence

    def sentence_EMB(self, a=0.0001):
        '''
        使用SIF方法获得加权句向量 注: 此处未进行SVD
        -------------------------------------
        a: float (可调参数)
        :rtype: np.array
        '''
        clean_sentence = self._preprocess_sentence()
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

        if len(clean_sentence) == 0:  # if the sentence is empty after stopwords-removal
            vs = np.random.rand(self.model.wv.vector_size)
            return vs

        vs = vs/len(clean_sentence)

        return vs

    def do_SVD(self, VS):
        '''
        对整个文章的句向量进行SVD，每一个句向量减去第一个特征向量
        ------------------------------------------------
        VS: np.array
        '''
        [U,_,_] = np.linalg.svd(VS)
        self.singular_vector = U

    def extract(self, text, title=None):
        '''
        使用模型从输入文本中提取摘要内容。
        ------------------------------------
        model: 模型
        text: 输入文本
        title: 输入标题
        '''

        Matrix = np.empty((0, self.word_vector_size))
        # Seperate sentences
        sentences = text2sentence(text)
        if not title and sentences:
            # Set the first sentence as the title, if no title
            title = sentences[0]
        # TODO Determine a suitable way to set the output length
        output_length = int(len(sentences) * self.abstract_percent)
        # Add a length limit
        if output_length > self.max_output_length:
            output_length = self.max_output_length

        # If there is less than 3 sentences in article, return the 1st sentence as abstract
        if len(sentences) <= 3:
            return sentences[0]

        ## Get sentence embeddings of every sentence
        Vs = {}
        for sentence in sentences:
            self.load_data(sentence)
            vs = self.sentence_EMB(self.sentence_embed_a)
            Matrix = np.vstack((Matrix,vs))
            Vs[sentence] = vs
        Matrix = np.transpose(Matrix)
        self.do_SVD(Matrix)
        u = self.singular_vector[:,0].reshape(self.word_vector_size,1)
        PC = u * np.transpose(u)
        for sentence in sentences:
            vs = Vs[sentence]
            vs = vs-PC.dot(vs)
            Vs[sentence] = vs

        del Matrix

        ## Get sentence embeddings of title
        self.load_data(title)
        Vt = self.sentence_EMB(self.sentence_embed_a)
        Vt = Vt-PC.dot(Vt)

        ## Get sentence embeddings of whole text
        self.load_data(text)
        Vc = self.sentence_EMB(self.sentence_embed_a)
        Vc = Vc-PC.dot(Vc)

        ## give score to every sentence and do KNN smoothins
        Score_dict = Get_Score(Vs,Vt,Vc,w1=self.score_title_weight,w2=1-self.score_title_weight)
        Score_dict = do_KNN(Score_dict,sentences,w_C=self.knn_w_c, w_O=self.knn_w_o, k=self.knn_k)

        ## generate output summarization and save it into file
        output = summarize(Score_dict,sentences,k=output_length)

        return ' '.join(output)
