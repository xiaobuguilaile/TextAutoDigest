# -*-coding:utf-8 -*-

'''
@File       : fasttext_embeddings.py
@Author     : HW Shen
@Date       : 2020/3/11
@Desc       :
'''


import gensim
from gensim.models import word2vec, fasttext
import pandas as pd
import loguru


class FastTextEmbedding:

    def __init__(self):
        '''
        初始化fasttext
        '''
        self.fast_model = None

    def train(self, data, size=100, window=10, min_count=5, workers=4):
        '''
        训练词向量
        ----------
        data: 数据或路径，传入路径时需要对应数据中各词按空格分隔
        size: 词向量长度（默认100）
        window: 上下文窗口大小（默认5）
        min_count: 忽略出现次数小于min_count的词语（默认5）
        workers: 训练使用线程数（默认3）
        '''
        # 判断data是否是路径
        if isinstance(data, str):
            # 读取数据
            data = word2vec.LineSentence(data)
        # 训练模型
        logger.info("词向量训练开始...")
        logger.info("参数[size:{},windows:{},min_count:{},workers:{}]".format(size, window, min_count, workers))
        self.fast_model = fasttext.FastText(data, size=size, window=window, min_count=min_count, workers=workers)
        logger.info("训练完毕")

    def save_model(self, path):
        '''
        保存已训练好的词向量
        --------------------
        path: 保存模型的路径
        '''
        if self.fast_model:
            self.fast_model.wv.save_word2vec_format(path, binary=True)
            logger.info("'{}'保存成功！".format(path))
        else:
            logger.info("词向量未训练！")

    def load_model(self, path):
        '''
        获取预先训练的词向量模型
        --------------------
        path: 保存模型的路径
        '''
        self.fast_model = fasttext.FastText.load_fasttext_format(path)

    def get_similar(self, word):
        '''
        获取"近义词most_similar"
        ----------
        word: 目标词语
        '''
        if not self.fast_model:
            logger.info("模型不存在！")
        if word not in self.fast_model.wv:
            logger.info("'{}' 不在词库中！".format(word))
        return self.fast_model.wv.most_similar(word)

    def get_vec(self, words):
        '''
        获取词语对应向量
        ----------------
        words: 目标词列表
        '''
        if not self.fast_model:
            logger.info("模型不存在！")
        for word in words:
            # TODO 处理不在词库中的词语，返回全零或特定向量
            if word not in self.fast_model.wv:
                logger.info("'{}' 不在词库中！".format(word))
                return None
        return self.fast_model.wv[words]

    def evaluate_model(self, path, test_words):
        """
        评估不同参数下fasttext_model的表现情况
        :return:
        """
        self.load_model(path)
        for word in test_words:
            try:
                logger.info(word + ": " + str(self.get_similar(word)))
            except Exception as e:
                pass


def function():

    # fast2vec模型的测试结果
    test_words = ["北京", "华为", "自然语言处理", "神经网络", "孙杨", "感受视野", "噌吰", "新冠疫情", "向量", "矩阵"]
    logger.info("test words: " + str(test_words))
    logger.info("----------------------------")
    # 加载bin格式的模型
    model_path = "../outputs/fasttext_s100_w10.bin"
    fast_model = fast.load_model(model_path)
    for word in test_words:
        try:
            logger.info(word + ": " + str(fast_model.most_similar(word)))
        except Exception as e:
            logger.info(e)


if __name__ == '__main__':

    # 设置日志输出
    logger = loguru.logger
    logger.add("../log/w2v/w2v_model_eveluation_{time}.log", encoding='utf-8')

    fast = FastTextEmbedding()
    data_path = "../data/combined_preprocess_data(re_stopwords).csv"
    fast.train(data_path)
    model_path = "../outputs/fasttext_s100_w10.bin"
    fast.save_model(model_path)

    function()

