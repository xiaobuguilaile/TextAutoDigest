# -*-coding:utf-8 -*-

'''
@File       : arg_util.py
@Author     : TY Liu
@Date       : 2020/3/19
@Desc       : 启动参数处理
'''

import argparse

def parse_args():
    '''
    获取启动参数并进行处理
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="./data")
    parser.add_argument('--wv_file_name', type=str, default="word_vectors_100")
    parser.add_argument('--score_title_weight', type=float, default=0.4)
    parser.add_argument('--knn_w_c', type=float, default=2)
    parser.add_argument('--knn_w_o', type=float, default=5)
    parser.add_argument('--knn_k', type=int, default=2)
    parser.add_argument('--sentence_embed_a', type=float, default=1e-4)
    parser.add_argument('--abstract_percent', type=float, default=0.2)
    parser.add_argument('--max_output_length', type=float, default=20)
    parser.add_argument('--debugging', type=int, default=0)
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed
