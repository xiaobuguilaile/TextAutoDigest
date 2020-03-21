# 设置参数
# 词向量文件所在目录
data_dir=./data
# 词向量文件名
wv_file_name=word_vectors_100
# 最终得分计算中标题的加权，全文的加权为1-score_title_weight
score_title_weight=0.4
# knn计算，上下文(context)的权重
knn_w_c=2
# knn计算，中心句(observation)的权重
knn_w_o=5
# knn计算，近邻个数
knn_k=2
# 使用SIF方法获得加权句向量的权重参数
sentence_embed_a=1e-4
# 默认获取摘要占全文的百分比
abstract_percent=0.2
# 最大摘要句子数目
max_output_length=20
# 是否开启debug模式(0:关闭，1：开启)
debugging=0

# 运行python服务
echo "############ Hebron Starting... #################"
python ./application.py --data_dir=$data_dir \
                        --wv_file_name=$wv_file_name \
                        --score_title_weight=$score_title_weight \
                        --knn_w_c=$knn_w_c \
                        --knn_w_o=$knn_w_o \
                        --knn_k=$knn_k \
                        --sentence_embed_a=$sentence_embed_a \
                        --abstract_percent=$abstract_percent \
                        --max_output_length=$max_output_length \
                        --debugging=$debugging