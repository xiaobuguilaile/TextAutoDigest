### 文本摘要

#### 数据预处理
要求训练集和测试集分开存储，对于中文的数据必须先分词，对分词后的词用空格符分开
* eg. 今天 的 天气 真好

#### 文件结构介绍
* config文件：配置各种模型的配置参数
* data：存放原始数据raw data，停用词stopwords
* preprocess：提供数据预处理的方法
* outputs：存放 vocab，word_to_index, label_to_index 处理后的数据
* models：存放模型代码
* trainers：存放训练代码
* predictors：存放预测代码