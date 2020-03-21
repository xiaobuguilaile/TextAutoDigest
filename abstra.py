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
from utils.arg_util import parse_args
import logging
app = Flask(__name__)

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
    path = os.path.abspath(FLAGS.data_dir+"/"+FLAGS.wv_file_name)
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
    app.run()


