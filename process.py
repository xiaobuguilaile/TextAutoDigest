import re


def fila(chuan):
    lili = re.split('。|!|；|？', chuan)
    return lili[0] + '。' + lili[-2] + '。'
