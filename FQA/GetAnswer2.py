from sklearn.feature_extraction.text import TfidfVectorizer #文本特征值提取划分
from sklearn.neighbors import KNeighborsClassifier  #KNN算法
from sklearn.decomposition import PCA#pca降维
from sklearn.model_selection import train_test_split #划分数据集 分成训练集和测试集
from sklearn.preprocessing import StandardScaler  #标准化
import jieba  #用来分词
import math
import numpy as np
import json
from build_vocab import Vocab
import pickle

'''
with open('qa.json', encoding='utf-8') as f:
    questions=[]
    answers=[]
    for i in range(1000):
        line = f.readline()
        d = json.loads(line)
        questions.append(d['question'])
        answers.append(d['answers'][0])
'''



questions=list(np.loadtxt(r"Q.txt",encoding='utf-8',dtype='str'))
answers=list(np.loadtxt(r"A.txt",encoding='utf-8',dtype='str'))
stop_words=list(np.loadtxt(r"stop_words.txt",encoding='utf-8',dtype='str'))
#记录问题答案号
# index=[i for i in range(len(answers))]
#分词
k=[]
for q in questions:
    k.append(' '.join(list(jieba.cut(q))))

# def data_split(question,answer_index,train,validation,test):
#     x_train, x_test, y_train, y_test = train_test_split(question,answer_index,test_size=1-train, random_state=100)
#     if(train<1):
#         mid=test/(1-train)
#         x_val,x_test,y_val,y_test=train_test_split(x_test,y_test,test_size=mid,random_state=100)
#         return x_train,x_val,x_test,y_train,y_val,y_test
#     x_val=[]
#     y_val=[]
#     return x_train, x_val, x_test, y_train, y_val, y_test

question_train=k
#将语句向量化
Tf = TfidfVectorizer(stop_words=stop_words)
question_train = Tf.fit_transform(question_train).toarray()
#标准化
standard=StandardScaler()
question_train=standard.fit_transform(question_train)
#PCA降维
pca = PCA(n_components=0.8)
question_train = pca.fit_transform(question_train)
def distance(a,b):
    if(len(a)!=len(b) and len(a)!=0):
        return 10000
    dis=0
    for i in range(len(a)):
        dis+=(a[i]-b[i])**2
    return math.sqrt(dis)
def get_answer(question):
    question=[' '.join(list(jieba.cut(question)))]
    question = Tf.transform(question).toarray()
    question = standard.transform(question)
    question = pca.transform(question)
    score=[]
    for my_questions in question_train:
        score.append(distance(question[0],my_questions))
    return answers[score.index(min(score))],min(score)

class Vocab:
    UNK_TAG = "<UNK>"  # 表示未知字符
    PAD_TAG = "<PAD>"  # 填充符
    PAD = 0
    UNK = 1

    def __init__(self):
        self.dict = {  # 保存词语和对应的数字
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }
        self.count = {}  # 统计词频的

    def fit(self, sentence):
        """
        接受句子，统计词频
        :param sentence:[str,str,str]
        :return:None
        """
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1  # 所有的句子fit之后，self.count就有了所有词语的词频

    def build_vocab(self, min_count=1, max_count=None, max_features=None):
        """
        根据条件构造 词典
        :param min_count:最小词频
        :param max_count: 最大词频
        :param max_features: 最大词语数
        :return:
        """
        if min_count is not None:
            self.count = {word: count for word, count in self.count.items() if count >= min_count}
        if max_count is not None:
            self.count = {word: count for word, count in self.count.items() if count <= max_count}
        if max_features is not None:
            # [(k,v),(k,v)....] --->{k:v,k:v}
            self.count = dict(sorted(self.count.items(), lambda x: x[-1], reverse=True)[:max_features])

        for word in self.count:
            self.dict[word] = len(self.dict)  # 每次word对应一个数字

        # 把dict进行翻转
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence, max_len=None):
        """
        把句子转化为数字序列
        :param sentence:[str,str,str]
        :return: [int,int,int]
        """
        if len(sentence) > max_len:
            sentence = sentence[:max_len]
        else:
            sentence = sentence + [self.PAD_TAG] * (max_len - len(sentence))  # 填充PAD

        return [self.dict.get(i, 1) for i in sentence]

    def inverse_transform(self, incides):
        """
        把数字序列转化为字符
        :param incides: [int,int,int]
        :return: [str,str,str]
        """
        return [self.inverse_dict.get(i, "<UNK>") for i in incides]

    def __len__(self):
        return len(self.dict)
if __name__ == '__main__':
    # voc_model = pickle.load(open("./models/vocab.pkl", "rb"))
    pass
# res,score=get_answer("图书馆在哪儿")
# print(res,score)
# res,score=get_answer("图书馆在哪")
# print(res,score)


