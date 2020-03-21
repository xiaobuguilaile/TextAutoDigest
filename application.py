'''
@File       : abstra.py
@Author     : W Li, TY Liu
@Date       : 2020/3/13
@Desc       : Main entrance, using flask server.
'''

from flask import Flask, render_template, request, jsonify
import json
import os
from models.SIF_model import GETSentence_Embedding
import logging
import argparse

app = Flask(__name__)


def parse_args(check=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="./outputs")
    parser.add_argument('--wv_file_name', type=str, default="word_vectors_s100_w10.bin")
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


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/send_message/', methods=['POST'])
def send_message():
    # global message_get
    message_get = request.form['message']
    title_get = request.form['title']
    # message_get = request.args['message']
    app.logger.info("接收标题：%s" % title_get)
    abstract = model.extract(message_get, title_get)
    # TODO Inhence the output via syntax
    app.logger.info("最终摘要：%s" % abstract)
    message_json = {
        "code": 200,
        "message": abstract
    }
    return jsonify(message_json)


if __name__ == '__main__':

    # 获取启动参数
    FLAGS, unparsed = parse_args()
    if unparsed:
        # TODO 处理问题参数
        pass
    app.logger.info("Arguments parsed, model building...")

    # 词向量路径
    path = os.path.abspath(FLAGS.data_dir + "/" + FLAGS.wv_file_name)

    # 建立模型
    model = GETSentence_Embedding(path=path, 
                                  score_title_weight=FLAGS.score_title_weight, 
                                  knn_w_c=FLAGS.knn_w_c, 
                                  knn_w_o=FLAGS.knn_w_o, 
                                  knn_k=FLAGS.knn_k, 
                                  sentence_embed_a=FLAGS.sentence_embed_a,
                                  abstract_percent=FLAGS.abstract_percent, 
                                  max_output_length=FLAGS.max_output_length)
    app.logger.info("Model building succeed!")
    app.logger.info("############ Ready to Serve #################")

    # 启动服务
    # 设置debug模式
    app.debug = FLAGS.debugging
    app.run(host='192.168.2.97', port=5000)


