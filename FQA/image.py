# _*_ coding: utf-8 _*_
import requests
# from build_vocab import Vocab
import pickle
import re

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
def img(url):
    im = requests.get(url).content
    url = 'https://www.how-old.net/Home/Analyze?isTest=False&source=&version=www.how-old.net'
    data = {'file': im}
    # url = url.encode()
    r = requests.post(url, files=data).content.replace('\\'.encode(), ''.encode())

    gender = re.search(r'"gender": "(.*?)"rn'.encode(), r)
    age = re.search(r'"age": "(.*?)"rn'.encode(), r)
    # if gender.group(1) == 'Male':
    #     gender = '男'
    # else:
    #     gender = '女'
    datas = [gender, age]
    return datas
# voc_model = pickle.load(open("./models/vocab.pkl", "rb"))
# url = "http://mmbiz.qpic.cn/mmbiz_jpg/MJQF4qpYRI2iagOx338eERia1XbS3kIG0W75kiaicAFianLn4icUumLpIDZXFLZGFYL9XN0gkfK0JhvCMyAgrBBfIPUQ/0"
# print(img('http://mmbiz.qpic.cn/mmbiz_jpg/MJQF4qpYRI2iagOx338eERia1XbS3kIG0W75kiaicAFianLn4icUumLpIDZXFLZGFYL9XN0gkfK0JhvCMyAgrBBfIPUQ/0'
#
#           ))