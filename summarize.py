from models.SIF_model import GETSentence_Embedding,text2sentence,Get_Score,sigmoid,do_KNN,summarize
import os
import numpy as np
import sys

if __name__ == '__main__':
    """
    this script generates summarization of text based on SIF 
    usage:
    python3 summarize.py <textpath> <titlepath> <outputlength>
    e.g.:
    python3 summarize.py ./data/text.txt ./data/title.txt 10
    """
    if len(sys.argv) != 4:
        print("this script generates summarization of text based on SIF \n\nusage:python3 summarize.py <textpath> <titlepath> <outputlength> \n\ne.g.:python3 summarize.py ./data/text.txt ./data/title.txt 10")
        sys.exit(0)
    path = os.path.abspath("./data/word_vectors_100")
    output_length = int(sys.argv[3])
    
    
    Calc_SE = GETSentence_Embedding(path)
    ft = open(os.path.abspath(sys.argv[1]),'r',encoding='utf-8')
    text = ft.read()
    ft.close()
    fti = open(os.path.abspath(sys.argv[2]),'r',encoding='utf-8')
    title = fti.read()
    fti.close()

    ## Get sentence embeddings of every sentence
    Matrix = np.empty((0,100))
    sentences = text2sentence(text)
    if output_length > len(sentences):
        raise ValueError("the summarize length must smaller than the length of original text!")
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
    Score_dict = Get_Score(Vs,Vt,Vc,w1=0.4,w2=0.6)
    Score_dict = do_KNN(Score_dict,sentences,k=2)

    ## generate output summarization and save it into file
    output = summarize(Score_dict,sentences,k=output_length)

    f = open(os.path.abspath('./data/summary.txt'),'w',encoding='utf-8')
    f.write(' '.join(output))
    f.close()
    