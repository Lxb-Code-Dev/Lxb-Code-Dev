# -*-coding:utf-8-*-
import os
import pickle
import re
import zipfile

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# from build_vocab import Vocab


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

class ImdbDataset(Dataset):
    def __init__(self, train=True):
        super(ImdbDataset,self).__init__()
        filepath='train.txt' if train else 'test.txt'
        self.filepath=filepath
        self.train = []
        f = open(filepath, "r", encoding='utf-8')
        self.train = f.readlines()
        self.train = self.train[1:]
        self.train_label = []
        self.train_data = []
        for sentence in self.train:
            temp = sentence.split(',')
            self.train_label.append(temp[0])
            temp[1] = temp[1].replace('\n', '')
            self.train_data.append(temp[1].split(" "))
    def __getitem__(self, idx):
        review=self.train_data[idx]
        label=self.train_label[idx]
        label=0 if label=='negative' else 1
        return review, label

    def __len__(self):
        return len(self.train)


def collate_fn(batch):
    """
    对batch数据进行处理
    :param batch: [一个getitem的结果，getitem的结果,getitem的结果]
    :return: 元组
    """
    reviews, labels = zip(*batch)
    return reviews, labels



if __name__ == "__main__":
    # vocab_model = pickle.load(open("./models/vocab.pkl", "rb"))
    pass
    # pass
    # imdb_dataset = ImdbDataset(True)
    # my_dataloader = DataLoader(imdb_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    # for data in my_dataloader:
        # vocab_model = pickle.load(open("./models/vocab.pkl", "rb"))
        # print(data)
        # result = vocab_model.transform(data[0][0], 100)
        # print(result)
        # break

    # unzip_file("./data/a.zip", "./data/download")
    # if os.path.exists("./data/download"):
    #     print("T")

    # data = open("./data/download/train/pos\\10032_10.txt", "r", encoding="utf-8").read()
    # result = tokenlize("--or something like that. Who the hell said that theatre stopped at the orchestra pit--or even at the theatre door?")
    # result = tokenlize(data)
    # print(result)

    # test_file()