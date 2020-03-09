from flask import Flask, render_template, request, jsonify
from process import fila
app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/send_message/', methods=['GET'])
def send_message():
    global message_get
    message_get = ""

    message_get = request.args['message']
    message_get = fila(message_get)
    print("摘要：%s" % message_get)
    return '收到'


@app.route('/change_to_json/', methods=['GET'])
def change_to_json():

    global message_get
    message_json = {
        "message": message_get
    }

    return jsonify(message_json)


if __name__ == '__main__':
    app.run()


