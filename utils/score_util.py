# -*-coding:utf-8 -*-

'''
@File       : score_util.py
@Author     : TY Liu
@Date       : 2020/3/19
@Desc       : 摘要得分计算
'''

import numpy as np
import re
from nltk.translate.bleu_score import sentence_bleu
import difflib
from rouge import Rouge
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # 当前程序上上一级目录
sys.path.append(BASE_DIR)  # 添加环境变量
from models.SIF_model import GETSentence_Embedding, cleanReviewChinese

# 带有人工摘要内容的测试文本标题
test_set = {"攀钢钒钛承认子公司污染任志强微博令其陷环保危机","财经观察：房产税全面开征遥遥无期","亚光科技产品加速“军转民” 毫米波TR组件助力安检升级","苏宁发布“3·15”大数据"}

def get_mean_score(model, test_dir="../data/test/", stopwords_path="../data/stopwords.txt", verbose=1):
    '''
    对测试文本计算平均得分
    '''
    bleu = 0
    rouge_1 = 0
    rouge_2 = 0
    rouge_l = 0
    similarity = 0
    for title in test_set:
        # 原文
        with open(test_dir + title + "origin.txt", "r",encoding='utf-8') as f:
            origin_text = f.read()
        # 获取人工摘要结果
        with open(test_dir + title + ".txt", "r", encoding='utf-8') as f:
            label_text = f.read()
        if verbose:
            print(title)
        # 自动摘要
        text = model.extract(origin_text, title)
        # bleu
        bleu += get_bleu(text, label_text, stopwords_path)
        # rouge
        rouge_result = get_rouge(text, label_text, stopwords_path, verbose)
        rouge_1 += rouge_result[0]["rouge-1"]['f']
        rouge_2 += rouge_result[0]["rouge-2"]['f']
        rouge_l += rouge_result[0]["rouge-l"]['f']
        # similarity
        similarity += get_similarity(text, label_text)

    # 计算均值
    n = len(test_set)
    bleu /= n
    rouge_1 /= n
    rouge_2 /= n
    rouge_l /= n
    similarity /= n
    return bleu, rouge_1, rouge_2, rouge_l, similarity

def get_similarity(text, label_text):
    '''
    计算文本相似性
    --------------
    text: 自动摘要结果
    label_text: 人工摘要结果
    '''
    # 计算相似性
    matcher = difflib.SequenceMatcher(lambda x: x == " ", text, label_text)
    ratio = matcher.ratio()
    return ratio

def get_rouge(text, label_text, stopwords_path="../data/stopwords.txt", verbose=1):
    '''
    计算ROUGE得分
    --------------
    text: 自动摘要结果
    label_text: 人工摘要结果
    '''
    textRouge = Rouge()
    text = " ".join(cleanReviewChinese(text, stopwords_path))
    label_text = " ".join(cleanReviewChinese(label_text, stopwords_path))
    rouge_score = textRouge.get_scores(text, label_text)
    if verbose:
        print(rouge_score[0]["rouge-1"])
        print(rouge_score[0]["rouge-2"])
        print(rouge_score[0]["rouge-l"])
    return rouge_score

def get_bleu(text, label_text, stopwords_path="../data/stopwords.txt"):
    '''
    计算BLEU得分
    ------------
    text: 自动摘要结果
    label_text: 人工摘要结果
    '''
    # 计算BLEU得分
    reference = cleanReviewChinese(label_text, stopwords_path)
    candidate = cleanReviewChinese(text, stopwords_path)
    bleu_score = sentence_bleu([reference], candidate)
    return bleu_score

if __name__ == '__main__':
    # 停用词路径
    stopwords_path="../data/stopwords.txt"
     # 获取启动参数
    FLAGS, unparsed = parse_args()
    print("Arguments parsed, model building...")
    # 词向量路径
    path = os.path.abspath(FLAGS.data_dir+"/"+FLAGS.wv_file_name)
    # 建立模型
    model = GETSentence_Embedding(path=path, 
                                  score_title_weight=FLAGS.score_title_weight, 
                                  knn_w_c=FLAGS.knn_w_c, 
                                  knn_w_o=FLAGS.knn_w_o, 
                                  knn_k=FLAGS.knn_k, 
                                  sentence_embed_a=FLAGS.sentence_embed_a,
                                  abstract_percent=FLAGS.abstract_percent, 
                                  max_output_length=FLAGS.max_output_length,
                                  stopwords_path=stopwords_path)

    print("Model building succeed!")
    print("############ Ready to Serve #################")
    bleu, r1, r2, rl, sim = get_mean_score(model)
    res = "BLEU:"+str(bleu),"Rouge-1:"+str(r1),"Rouge-2:"+str(r2),"Rouge-l:"+str(rl),"Similarity:"+str(sim)
    print(res)
