from models.SIF_model import GETSentence_Embedding,text2sentence
import os
import numpy as np

def Get_Score(Vs,Vt,Vc,w1=0.05,w2=0.05):
    Score_dict = {}
    for (sentence,vs) in Vs.items():
        Score_dict[sentence] = sigmoid(w1*vs.dot(Vt)+w2*vs.dot(Vc))
    return Score_dict

def sigmoid(x):
    return 1/(1+np.exp(-x))

if __name__ == '__main__':
    cwd = os.getcwd()
    path = cwd+"/data/word_vectors_100"
    # jieba.load_userdict(cwd+"/utils/jieba_latest_dict.txt")
    
    
    Calc_SE = GETSentence_Embedding(path)
    ## TODO: 读取文章和标题
    text = "[环球网综合报道]21日15时许，国务院联防联控机制召开新闻发布会，介绍依法防控疫情、维护社会稳定工作情况。有记者提问，在这次疫情防控中出现了一些公民不文明的行为，比如向电梯吐口水或者公众场合不戴口罩、不听劝阻辱骂工作人员等，疫情期间对这种行为怎么定性？有什么处罚措施？公安部治安管理局局长李京生表示，疫情期间，我们发现个别人为了寻求刺激或者发泄不满，向公众、电梯、超市商品甚至医务人员吐口水，还有一些人拒不配合疫情防控，不采取任何防护措施进入公共场所且不听劝阻，辱骂殴打工作人员。这些“不文明”行为，在全国疫情防控的非常时期，增加了疫情隐患，加剧了恐慌气氛，甚至妨害疫情防控，危害公共安全。李京生称，对此，公安部高度重视，部署各地公安机关严格规范公正文明执法，依法查处打击。按照有关法律规定，疫情防控期间，在公共场所起哄闹事，造成公共场所秩序严重混乱的;明知感染或者疑似感染，故意进入公共场所或者交通工具传播的;对医务人员实施吐口水的行为，致使医务人员感染的;以暴力、威胁等方法阻碍国家机关工作人员和其他工作人员依法履行防疫、检疫、强制隔离、隔离治疗等措施的，构成犯罪的，依法追究刑事责任，不构成犯罪的，依法处以治安处罚。此外，对在公共场所未佩戴口罩的人员，公安民警将做好提醒和劝导等工作。在此，我们也提醒广大群众主动支持、积极配合落实各项防疫措施，共同维护疫情期间的各项社会秩序。他介绍，对妨害疫情防控的各类违法犯罪行为，公安机关将依法予以严厉打击，截至目前，各地公安机关累计查处哄抬物价、囤积居奇等非法经营案件274起，非法收购、运输、出售野生动物案件1787起，以危险方法危害公共安全案件198起，妨害传染病防治案件2530起，妨害公务案件3446起，利用疫情实施诈骗案件3013起，编造、故意传播虚假及有害信息案件5511起，制售假劣口罩等防护物资案件604起，制售涉疫伪劣产品、假劣药品、医疗器械、医用卫生材料案件722起。累计刑事拘留3644人，行政拘留2.5万名，批评教育4.6万名。"
    title = "向电梯吐口水、不戴口罩、辱骂工作人员等行为，如何定性？公安部回应"

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

    Score_dict = Get_Score(Vs,Vt,Vc)
    sorted_sent = sorted(Score_dict.items(),key=lambda x: x[1],reverse=True)
    print(sorted_sent)
    