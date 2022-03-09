# -*-coding:utf-8-*-
import pickle
import torch
import os
import jieba
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import dataset
import numpy
from build_vocab import Vocab
from torch.utils.data import Dataset, DataLoader
from prepro import *

# class Vocab:
#     UNK_TAG = "<UNK>"  # 表示未知字符
#     PAD_TAG = "<PAD>"  # 填充符
#     PAD = 0
#     UNK = 1
#
#     def __init__(self):
#         self.dict = {  # 保存词语和对应的数字
#             self.UNK_TAG: self.UNK,
#             self.PAD_TAG: self.PAD
#         }
#         self.count = {}  # 统计词频的
#
#     def fit(self, sentence):
#         """
#         接受句子，统计词频
#         :param sentence:[str,str,str]
#         :return:None
#         """
#         for word in sentence:
#             self.count[word] = self.count.get(word, 0) + 1  # 所有的句子fit之后，self.count就有了所有词语的词频
#
#     def build_vocab(self, min_count=1, max_count=None, max_features=None):
#         """
#         根据条件构造 词典
#         :param min_count:最小词频
#         :param max_count: 最大词频
#         :param max_features: 最大词语数
#         :return:
#         """
#         if min_count is not None:
#             self.count = {word: count for word, count in self.count.items() if count >= min_count}
#         if max_count is not None:
#             self.count = {word: count for word, count in self.count.items() if count <= max_count}
#         if max_features is not None:
#             # [(k,v),(k,v)....] --->{k:v,k:v}
#             self.count = dict(sorted(self.count.items(), lambda x: x[-1], reverse=True)[:max_features])
#
#         for word in self.count:
#             self.dict[word] = len(self.dict)  # 每次word对应一个数字
#
#         # 把dict进行翻转
#         self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))
#
#     def transform(self, sentence, max_len=None):
#         """
#         把句子转化为数字序列
#         :param sentence:[str,str,str]
#         :return: [int,int,int]
#         """
#         if len(sentence) > max_len:
#             sentence = sentence[:max_len]
#         else:
#             sentence = sentence + [self.PAD_TAG] * (max_len - len(sentence))  # 填充PAD
#
#         return [self.dict.get(i, 1) for i in sentence]
#
#     def inverse_transform(self, incides):
#         """
#         把数字序列转化为字符
#         :param incides: [int,int,int]
#         :return: [str,str,str]
#         """
#         return [self.inverse_dict.get(i, "<UNK>") for i in incides]
#
#     def __len__(self):
#         return len(self.dict)
train_batch_size = 512
test_batch_size = 128

sequence_max_len = 100

voc_model = pickle.load(open("./models/vocab.pkl", "rb"))






def collate_fn(batch):
    """
    对batch数据进行处理
    :param batch: [一个getitem的结果，getitem的结果,getitem的结果]
    :return: 元组
    """
    reviews, labels = zip(*batch)
    reviews = torch.LongTensor([voc_model.transform(i, max_len=sequence_max_len) for i in reviews])
    labels = torch.LongTensor(labels)
    return reviews, labels


def get_dataloader(train=True):
    imdb_dataset = dataset.ImdbDataset(train)
    batch_size = train_batch_size if train else test_batch_size
    return DataLoader(imdb_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)






class ImdbDataset2(Dataset):
    def __init__(self,file_p=''):
        super(ImdbDataset2,self).__init__()
        self.filepath=file_p
        filepath = file_p
        self.train = []
        f = open(filepath, "r", encoding='utf-8')
        self.train = f.readlines()
        # self.train = self.train[1:]
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
class ImdbModel(nn.Module):
    def __init__(self):
        super(ImdbModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(voc_model), embedding_dim=200, padding_idx=voc_model.PAD).to()
        self.lstm = nn.LSTM(input_size=200, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True,
                            dropout=0.5)
        self.fc1 = nn.Linear(64 * 2, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, input):
        """
        :param input:[batch_size,max_len]
        :return:
        """
        input_embeded = self.embedding(input)  # input embeded :[batch_size,max_len,200]

        output, (h_n, c_n) = self.lstm(input_embeded)  # h_n :[4,batch_size,hidden_size]
        # out :[batch_size,hidden_size*2]
        out = torch.cat([h_n[-1, :, :], h_n[-2, :, :]], dim=-1)  # 拼接正向最后一个输出和反向最后一个输出

        # 进行全连接
        out_fc1 = self.fc1(out)
        # 进行relu
        out_fc1_relu = F.relu(out_fc1)

        # 全连接
        out_fc2 = self.fc2(out_fc1_relu)  # out :[batch_size,2]
        return F.log_softmax(out_fc2, dim=-1)


def device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def train(imdb_model, epoch):
    """

    :param imdb_model:
    :param epoch:
    :return:
    """
    train_dataloader = get_dataloader(train=True)


    optimizer = Adam(imdb_model.parameters(),lr=0.005)
    for i in range(epoch):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        for idx, (data, target) in enumerate(bar):
            optimizer.zero_grad()
            data = data.to(device())
            target = target.to(device())
            output = imdb_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            bar.set_description("epcoh:{}  idx:{}   loss:{:.6f}".format(i, idx, loss.item()))


def test(imdb_model):
    """
    验证模型
    :param imdb_model:
    :return:
    """
    test_loss = 0
    correct = 0
    imdb_model.eval()
    test_dataloader = get_dataloader(train=False)
    with torch.no_grad():
        for data, target in tqdm(test_dataloader):
            data = data.to(device())
            #target是标签，data是句子列表为单元的数组，长度都等于test_batch_size，都是Tensor类型
            target = target.to(device())
            output = imdb_model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]  # 获取最大值的位置,[batch_size,1]
            #下边这行target.data与target为相同张量，多此一举了
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_dataloader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)))


def test_mymodel(file):
    '''
    :param file: 一个txt文件路径，该txt文件存储了一天之内该用户所有的聊天内容
    并且将该文件删除
    :return: 一个字符串,positive或者negative或者None，返回该用户一天情绪的综合指标,
    其中，None表示没有聊天
    '''
    #判断文件是否为空
    size = os.path.getsize(file)
    if size==0:
        return 'None'

    imdb_model = torch.load('./models/temp')
    #加载模型
    #首先需要进行分词处理
    word_segmentation(file)
    total_all = 0
    #计算所有的语句之和
    handle = open('talks.txt', 'r',encoding='utf-8')
    for eachline in handle:
        total_all += 1
    imdb_dataset = ImdbDataset2(file)
    test_dataloader = DataLoader(imdb_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    imdb_model.eval()
    #不能让模型改变
    total = 0
    #total表示所有积极的语句
    with torch.no_grad():
        for data, target in tqdm(test_dataloader):
            data = data.to(device())
            # target是标签，data是句子列表为单元的数组，长度都等于test_batch_size，都是Tensor类型
            output = imdb_model(data)
            pred = output.data.max(1, keepdim=True)[1]  # 获取最大值的位置,[batch_size,1]
            # if temp==0:
            x = pred.cpu()
            a = x.numpy()
            a = numpy.sum(a)
            total = total+a

    #此处设定阈值为50%，也就是说，消极的占到总语句的一半就反馈为negative
    if((total_all-total)/total_all>0.5):
        os.remove(file)
        return 'negative'
    else:
        os.remove(file)
        return 'positive'

#用于分词函数
def word_segmentation(input):
    new_input = ' '.join(jieba.cut(input))
    return new_input
def word_segmentation(filepath):
    '''
    :param filepath: 接收一个file路径
    :return: 将其中的语句分词，改变该文件
    该函数假设filepath都是有内容的
    '''
    f = open(filepath, "r", encoding='utf-8')
    f2 = open(filepath+'temp', "a", encoding='utf-8')
    for sentence in f:
        temp = sentence.split(',')
        new_text = ' '.join(jieba.cut(temp[1]))
        f2.write(temp[0]+','+new_text)
    f.close()
    f2.close()
    os.remove(filepath)
    os.rename(filepath+'temp',filepath)

record = dict()
#计数，每7天一更新数据
seven = 0
def sentiment_ana():
    attention = []
    #attention用于返回信息,为空列表说明没有预警，否则返回的列表中是user的openID
    global seven
    seven+=1
    # 遍历message_log中的每个文件，
    file_list = os.listdir('message_log')
    for file in file_list:
        label = test_mymodel(file)
        if label == 'negative':
            # 如果找不到则返回0
            find = record.get(label, 0)
            if (find == 0):
                record[label] = 1
            else:
                record[label] += 1
    # 7天里边有至少5天消极，发出预警
    #还需要做的工作是进行清除所有文本
    if seven%7==0:
        for file in file_list:
            os.remove(file)
        for user in record:
            if record[user]>=5:
                attention.append(user)


    return attention
# if __name__ == '__main__':
#     pass
    # imdb_model = ImdbModel().to(device())
    # train(imdb_model, 8)
    # torch.save(imdb_model, './models/temp')

    # test(imdb_model)
    # test_mymodel()
    # test(my_model)
    # label = test_mymodel(my_model,'temp.txt')
    # print(label)