# -*-coding:utf-8 -*-

'''
@File       : w2v_embeddings.py
@Author      : TY Liu
@Date       : 2020/2/17
@Desc       :
'''

from gensim.models import word2vec
import pandas as pd

class Word2Vec:
    '''
    w2v词向量
    '''
    def __init__(self):
        '''
        初始化Word2Vec
        '''
        self.model = None
    
    def train(self, data, size=100, window=10, min_count=5, workers=4):
        '''
        训练词向量
        ----------
        data: 数据或路径，传入路径时需要对应数据中各词按空格分隔
        size: 词向量长度
        window: 上下文窗口大小
        min_count: 忽略出现次数小于min_count的词语
        workers: 训练使用线程数
        '''
        # 判断data是否是路径
        if type(data) == str:
            # 读取数据
            data = word2vec.LineSentence(data)
        # 训练模型
        print("词向量训练开始...")
        print("参数[size:{},windows:{},min_count:{},workers:{}".format(size,window,min_count,workers))
        self.model = word2vec.Word2Vec(data, size=size, window=window, min_count=min_count, workers=workers)
        print("训练完毕")
    
    def save_model(self, path):
        '''
        保存已训练好的词向量
        --------------------
        path: 保存模型的路径
        '''
        if self.model:
            self.model.save(path)
            print("'{}'保存成功！".format(path))
        else:
            print("词向量未训练！")
    
    def load_model(self, path):
        '''
        获取预先训练的词向量
        --------------------
        path: 保存模型的路径
        '''
        self.model = word2vec.Word2Vec.load(path)
    
    def get_similar(self, word):
        '''
        获取近义词
        ----------
        word: 目标词语
        '''
        if not self.model:
            print("词向量未训练！")
        if word not in self.model.wv:
            print("'{}' 不在词库中！".format(word))
        return self.model.wv.most_similar(word)
    
    def get_vec(self, words):
        '''
        获取词语对应向量
        ----------------
        words: 目标词列表
        '''
        if not self.model:
            print("词向量未训练！")
        for word in words:
            # TODO 处理不在词库中的词语，返回全零或特定向量
            if word not in self.model.wv:
                print("'{}' 不在词库中！".format(word))
                return None
        return self.model.wv[words]
    
    def get_model(self):
        '''
        获取w2v模型
        '''
        if not self.model:
            print("词向量未训练！")
        return self.model
    
if __name__ == '__main__':
    w2v = Word2Vec()
    data_path="data/combined_preprocess_data(re_stopwords).csv"
    w2v.train(data_path)
    
    print(w2v.get_similar("向量"))
    
    model_path="output/word_vectors_100"
    w2v.save_model(model_path)
    