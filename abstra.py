from flask import Flask, render_template, request, jsonify
import json
import os
from models.SIF_model import GETSentence_Embedding
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
    print("标题：%s" % title_get)
    abstract = model.extract(message_get, title_get)
    # TODO Inhence the output via syntax
    print("摘要：%s" % abstract)
    message_json = {
        "code": 200,
        "message": abstract
    }
    return jsonify(message_json)


# @app.route('/change_to_json/', methods=['GET'])
# def change_to_json():

#     global message_get
#     message_json = {
#         "message": message_get
#     }

#     return jsonify(message_json)


if __name__ == '__main__':
    path = os.path.abspath("./data/word_vectors_100")
    model = GETSentence_Embedding(path)
    app.run()


