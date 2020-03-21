# -*-coding:utf-8 -*-

'''
@File       : eval.py
@Author     : TY Liu
@Date       : 2020/3/20
@Desc       : 模型评价，参数调优
'''

import os
from models.SIF_model import GETSentence_Embedding
from utils.score_util import get_mean_score
import logging
import time
import pandas as pd

# 设置logger
logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.INFO)

def eval(model):
    '''
    进行模型评价，获取测试文本上的摘要
    ----------------------------
    model: 模型
    '''
    bleu, r1, r2, rl, sim = get_mean_score(model, test_dir="./data/test/", stopwords_path="./data/stopwords.txt", verbose=0)
    res = "BLEU:"+str(bleu), "Rouge-1:"+str(r1), "Rouge-2:"+str(r2), "Rouge-l:"+str(rl),"Similarity:"+str(sim)
    print(res)
    logger.info(res)
    return bleu, r1, r2, rl, sim

def single_tuning(param, model):
    '''
    单个参数进行调整和评估
    --------------------
    param: 目标参数名
    model: 模型
    '''
    df = pd.DataFrame(columns=["param","value","bleu","rouge-1","rouge-2","rouge-3","similarity"])
    for p in params[param]:
        setattr(model, param, p)
        logger.info("======== eval "+param+"="+str(p)+"========")
        bleu, r1, r2, rl, sim = eval(model)
        df.loc[len(df)] = [param, p, bleu, r1, r2, rl, sim]
    return df

def double_tuning(param_a, param_b, model):
    '''
    双参数进行调整和评估
    -------------------
    param_a: 目标参数A
    param_b: 目标参数B
    model: 模型
    '''
    df = pd.DataFrame(columns=["param_a","value_a","param_b","value_b","bleu","rouge-1","rouge-2","rouge-3","similarity"])
    for p_a in params[param_a]:
        setattr(model, param_a, p_a)
        for p_b in params[param_b]:
            setattr(model, param_b, p_b)
            logger.info("======== eval "+param_a+"="+str(p_a)+" while "+param_b+"="+str(p_b)+" ========")
            bleu, r1, r2, rl, sim = eval(model)
            df.loc[len(df)] = [param_a, p_a, param_b, p_b, bleu, r1, r2, rl, sim] 
    return df

def triple_tuning(param_a, param_b, param_c, model):
    '''
    三参数进行调整和评估
    -------------------
    param_a: 目标参数A
    param_b: 目标参数B
    param_c: 目标参数C
    model: 模型
    '''
    df3 = pd.DataFrame(columns=["param_a","value_a","param_b","value_b","param_c","value_c","bleu","rouge-1","rouge-2","rouge-3","similarity"])
    for p_a in params[param_a]:
        setattr(model, param_a, p_a)
        for p_b in params[param_b]:
            setattr(model, param_b, p_b)
            for p_c in params[param_c]:
                setattr(model, param_c, p_c)
                logger.info("======== eval "+param_a+"="+str(p_a)+" while "+param_b+"="+str(p_b)+" and "+param_c+"="+str(p_c)+" ========")
                bleu, r1, r2, rl, sim = eval(model)
                df.loc[len(df)] = [param_a, p_a, param_b, p_b, param_c, p_c, bleu, r1, r2, rl, sim]


if __name__ == '__main__':
    # 设置log
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = os.path.dirname(os.getcwd()) + '/Logs/param_tuning_'
    log_name = log_path + rq + '.log'
    fh = logging.FileHandler(log_name, mode='w')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # 停用词路径
    stopwords_path="./data/stopwords.txt"
    # 词向量路径
    path = os.path.abspath("./data/word_vectors_100")
    # 建立模型
    model = GETSentence_Embedding(path=path,
                                  stopwords_path=stopwords_path)
    print("Model building succeed!")
    print("############ Ready to Eval #################")

    # 待调整参数列表
    params = dict()
    params['score_title_weight'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # 句子结果中Title的权重占比
    params['sentence_embed_a'] = [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1] # SIF 参数 a, 一般取1e-5 ~ 1e-3
    params['knn_w_c'] = [0.01, 0.1, 1, 10] # KNN-smooth 中 w-c(上下文)的权重值
    params['knn_w_o'] = [0.01, 0.1, 1, 10] # KNN-smooth 中 w-o(中心句)的权重值
    params['knn_k'] = [1,2,3,4] # KNN-smooth 中的 K
    params['abstract_percent'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] # 摘要占原文的百分比
    param_names = list(params.keys())

    # 单参数调整
    df1 = pd.DataFrame(columns=["param","value","bleu","rouge-1","rouge-2","rouge-3","similarity"])
    for param in param_names:
        origin = getattr(model, param)
        logger.info("Tuning >>>>>{0}<<<<<".format(param))
        df = single_tuning(param, model)
        df1 = pd.concat([df1, df], axis=0)
        setattr(model, param, origin)
    df1.to_csv("single_tuning.csv", index=False)

    # # 双参数调整
    # df2 = pd.DataFrame(columns=["param_a","value_a","param_b","value_b","bleu","rouge-1","rouge-2","rouge-3","similarity"])
    # for i, param_a in enumerate(param_names):
    #     origin_a = getattr(model, param_a)
    #     for param_b in param_names[i+1:]:
    #         origin_b = getattr(model, param_b)
    #         logger.info("Tuning >>>>>{0} && {1}<<<<<".format(param_a, param_b))
    #         df = double_tuning(param_a, param_b, model)
    #         df2 = pd.concat([df2, df], axis=0)
    #         setattr(model, param_b, origin_b)
    #     setattr(model, param_a, origin_a)
    # df2.to_csv("double_tuning.csv", index=False)

    # # 三参数调整
    # df3 = pd.DataFrame(columns=["param_a","value_a","param_b","value_b","param_c","value_c","bleu","rouge-1","rouge-2","rouge-3","similarity"])
    # for i, param_a in enumerate(param_names):
    #     origin_a = getattr(model, param_a)
    #     for j, param_b in enumerate(param_names[i+1:]):
    #         origin_b = getattr(model, param_b)
    #         for param_c in param_names[j+1:]:
    #             origin_c = getattr(model, param_c)
    #             logger.info("Tuning >>>>>{0} && {1} && {2}<<<<<".format(param_a, param_b, param_c))
    #             df = triple_tuning(param_a, param_b, param_c, model)
    #             df3 = pd.concat([df2, df], axis=0)
    #             setattr(model, param_c, origin_c)
    #         setattr(model, param_b, origin_b)
    #     setattr(model, param_a, origin_a)
    # df3.to_csv("triple_tuning.csv", index=False)

