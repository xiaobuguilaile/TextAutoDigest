# -*-coding:utf-8 -*-

'''
@File       : w2v_data_pre.py
@Author     : HW Shen
@Date       : 2020/2/14
@Desc       : word2vec模型的数据准备
'''

import jieba
jieba.load_userdict("../utils/jieba_latest_dict.txt")
from bs4 import BeautifulSoup
import pandas as pd
import re
from loguru import logger


def cleanReviewChinese(content):
    """
    中文文本预处理函数：去除各种标点符号，HTML标签，小写化
    :param content:
    :return:
    """
    # 去除HTML标签
    beau = BeautifulSoup(content, features="lxml")
    btext = beau.get_text()

    # 去除标点,数字, 字母及特殊字符
    sub_text = re.sub('[a-zA-Z0-9'
                      '１２３４５６７８９０'
                      'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ'
                      'ｅｓｎｇｒｉｋｄｃｈａｔｖｘｙｚｏｌｍｐｆｕ'
                      '!；：．·、…【】《》“”‘’！？"#$%&％\'?@，。〔〕［］（）()*+,-——\\./:;<=>＋×／'
                      '①↑↓★▌▲●℃[\\]^_`{|}~\s]+',
                      "",
                      btext)

    # 去除不可见字符
    newContent = re.sub('[\001\002\003\004\005\006\007\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a]+', '', sub_text)

    # 以空格划分单词
    words = jieba.lcut(newContent, cut_all=False)

    # 去除停止词
    with open("../data/stopwords.txt", encoding="utf-8-sig") as f:
        stopwords = [line.strip() for line in f.readlines()]
    filter_words = [word for word in words if word not in stopwords]

    res = " ".join(filter_words)

    return res


def w2v_Hanyu_preprocess():
    """
    汉语语料中文文本的预处理
    :return:
    """

    with open("../data/raw/sqlResult_1558435(utf8).csv", encoding="utf-8") as f:
        corpus = f.read()

    pattern = re.compile(r'"[\d]+","[\S\s]*?","[\S\s]*?","([\S\s]*?)"')
    contents = pattern.findall(corpus)
    # print(len(contents))

    raw_data = pd.DataFrame(contents, columns=["content"])

    pre_data = raw_data["content"].apply(cleanReviewChinese)  # 数据清洗

    # 保存成csv文件
    pre_data.to_csv("../data/hanyu_preprocess_data(re_stopwords).csv", index=False)


def w2v_wiki_preprocess():
    """
    wikipedia中文语料处理第3步数据清洗
    >> 1.经过wikiexractor
    >> 2.繁体转简体
    >> 3.数据清洗
    :return:
    """
    with open("../data/raw/wiki_zh_simple.txt", encoding="utf-8") as f:
        contents = [line.strip() for line in f.readlines()]

    raw_data = pd.DataFrame(contents, columns=["content"])

    pre_data = raw_data["content"].apply(cleanReviewChinese)  # 数据清洗

    # 保存成csv文件
    pre_data.to_csv("../data/wiki_preprocess_data(re_stopwords).csv", index=False)


def combined_data():
    """ 将wiki和汉语语料预处理的结果合并到一个文件中 """

    import shutil
    shutil.copyfile('../data/wiki_preprocess_data(re_stopwords).csv', '../data/combined_preprocess_data(re_stopwords).csv')

    with open("../data/hanyu_preprocess_data(re_stopwords).csv", encoding='utf-8') as f:
        lines = [line for line in f.readlines()]

    fw = open('../data/combined_preprocess_data(re_stopwords).csv', 'a', encoding='utf-8')

    for item in lines:
        fw.write(item)

    fw.close()


if __name__ == '__main__':

    # w2v_Hanyu_preprocess()
    # w2v_wiki_preprocess()
    combined_data()
