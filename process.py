from models.SIF_model import GETSentence_Embedding, text2sentence,Get_Score, sigmoid,do_KNN,summarize
import numpy as np


def extract(model, text, title=None):
    '''
    使用模型从输入文本中提取摘要内容。
    ------------------------------------
    model: 模型
    text: 输入文本
    '''

    Matrix = np.empty((0,100))
    # Seperate sentences
    sentences = text2sentence(text)
    if not title and sentences:
        # Using the first sentence as the title
        title = sentences[0]
    # TODO Determine a suitable way to set the output length
    abstract_percent = 0.2
    output_length = int(len(sentences) * abstract_percent)
    # Add a length limit
    if output_length > 20:
        output_length = 20

    ## Get sentence embeddings of every sentence
    Vs = {}
    for sentence in sentences:
        model.load_data(sentence)
        vs = model.sentence_EMB()
        Matrix = np.vstack((Matrix,vs))
        Vs[sentence] = vs
    Matrix = np.transpose(Matrix)
    model.do_SVD(Matrix)
    u = model.singular_vector[:,0].reshape(100,1)
    for sentence in sentences:
        vs = Vs[sentence]
        PC = u*np.transpose(u)
        vs = vs-PC.dot(vs)
        Vs[sentence] = vs

    del Matrix

    ## Get sentence embeddings of title
    model.load_data(title)
    Vt = model.sentence_EMB()
    Vt = Vt-PC.dot(Vt)

    ## Get sentence embeddings of whole text
    model.load_data(text)
    Vc = model.sentence_EMB()
    Vc = Vc-PC.dot(Vc)

    ## give score to every sentence and do KNN smoothins
    Score_dict = Get_Score(Vs,Vt,Vc,w1=0.4,w2=0.6)
    Score_dict = do_KNN(Score_dict,sentences,k=2)

    ## generate output summarization and save it into file
    output = summarize(Score_dict,sentences,k=output_length)

    return ' '.join(output)


if __name__ == '__main__':

    pass