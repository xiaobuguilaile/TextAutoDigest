# -*-coding:utf-8 -*-

'''
@File       : word_util.py
@Author      : TY Liu
@Date       : 2020/2/22
@Desc       : 词语频率和权重计算
'''

import pandas as pd
from collections import Counter
import json


def get_word_count(data):
    '''
    获取词语出现次数
    ----------------
    data: DataFrame或csv路径, 每行数据为一篇文章以空格分隔的分词结果
    '''
    # 传入路径时读取数据
    if type(data) == str:
        data = pd.read_csv(data, header=None)
        print("File loaded...")
    counts = Counter()
    for index, row in data.iterrows():
        # 使用update 更新 Counter 计数
        counts.update(row[0].split())
        if index % 10000 == 0:
            print("{:.2f}% rows handled...".format(index/len(data)*100))
    print("100% rows handled...")
    return counts


def get_word_freq(counter):
    '''
    获取词语出现频率
    ----------------
    counter: Counter，包含各词语的出现次数
    '''
    freq = {}
    total = sum(counter.values()) # 词语总数
    c = len(counter) # 词语类别总数，平滑参数
    index = 0
    for k,v in counter.items():
        # 计算频率，拉普拉斯平滑
        freq[k] = (v + 1)/ (total + c)
        if index % 10000 == 0:
            print("{:.2f}% word frequencies handled...".format(index/c*100))
        index += 1
    print("100% rows handled...")
    return freq


def get_word_weight(freq, a=1e-3):
    '''
    SIF-获取词语权重字典
    ---------------------
    freq: dict(词语，出现频率)
    a: 权重参数，一般取1e-5 ~ 1e-3
    '''
    # 处理无效参数，改为不设置权重
    if a <= 0:
        a = 1.0
        
    word_weight = {}
    # weight = a / (a + p(w))
    for k,v in freq.items():
        word_weight[k] = a / (a + v)
    return word_weight


def word_weight_process(data_path, file_path, weight_param=1e-3):
    '''
    从分词后的句子csv中获取词语权重并保存
    -------------------------------------
    data_path: 句子csv文件路径
    file_path: 保存权重文件的路径
    weight_param: 计算SIF词语权重使用的参数
    '''
    counter = get_word_count(data_path)
    print("word count obtained...")
    
    freq = get_word_freq(counter)
    print("word frequencies obtained...")
    del counter
    
    weights = get_word_weight(freq, weight_param)
    print("word weights obtained...")
    del freq
    
    weights_dump = json.dumps(weights)
    with open(file_path, "w") as f:
        f.write(weights_dump)
    print("word weights were saved as {}".format(file_path))
        
def reload_weights(file_path):
    '''
    直接读取已处理的词语权重字典
    -----------------------------
    file_path: 保存权重文件的路径
    '''
    f=open(file_path)
    weights = f.read()
    weights = json.loads(weights)
    f.close()
    return weights


if __name__ == '__main__':
    data_path = "../data/combined_preprocess_data(re_stopwords).csv"
    file_path = "../data/word_weight.txt"
    weight_param = 1e-3
    word_weight_process(data_path, file_path, weight_param)