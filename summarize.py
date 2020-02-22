from models.SIF_model import GETSentence_Embedding,text2sentence,Get_Score,sigmoid,do_KNN,summarize
import os
import numpy as np
import sys

if __name__ == '__main__':
    cwd = os.getcwd()
    path = cwd+"/data/word_vectors_100"
    output_length = int(sys.argv[1])
    
    
    Calc_SE = GETSentence_Embedding(path)
    ## TODO: 读取文章和标题
    ft = open(cwd+"/data/text.txt",'r',encoding='utf-8')
    text = ft.read()
    ft.close()
    fti = open(cwd+"/data/title.txt",'r',encoding='utf-8')
    title = fti.read()
    fti.close()

    ## Get sentence embeddings of every sentence
    Matrix = np.empty((0,100))
    sentences = text2sentence(text)
    Vs = {}
    for sentence in sentences:
        Calc_SE.load_data(sentence)
        vs = Calc_SE.sentence_EMB()
        Matrix = np.vstack((Matrix,vs))
        Vs[sentence] = vs
    Matrix = np.transpose(Matrix)
    Calc_SE.do_SVD(Matrix)
    u = Calc_SE.singular_vector[:,0].reshape(100,1)
    for sentence in sentences:
        vs = Vs[sentence]
        PC = u*np.transpose(u)
        vs = vs-PC.dot(vs)
        Vs[sentence] = vs

    del Matrix

    ## Get sentence embeddings of title
    Calc_SE.load_data(title)
    Vt = Calc_SE.sentence_EMB()
    Vt = Vt-PC.dot(Vt)

    ## Get sentence embeddings of whole text
    Calc_SE.load_data(text)
    Vc = Calc_SE.sentence_EMB()
    Vc = Vc-PC.dot(Vc)

    ## give score to every sentence and do KNN smoothins
    Score_dict = Get_Score(Vs,Vt,Vc)
    Score_dict = do_KNN(Score_dict,sentences,k=3)

    ## generate output summarization and save it into file
    output = summarize(Score_dict,sentences,k=output_length)

    f = open(cwd+'/data/summary.txt','w',encoding='utf-8')
    f.write(' '.join(output))
    f.close()
    