# import tensorflow as tf
from bert_serving.client import BertClient
import numpy as np
from build_vocab import Vocab
from bert_serving.server.helper import get_args_parser
import pickle
from bert_serving.server import BertServer
# args = get_args_parser().parse_args(['-model_dir', 'D:\AppCentral\chinese_L-12_H-768_A-12',
#                                      '-port', '15555', # 客户端连接填入port和port_out参数
#                                      '-port_out', '15556',
#                                      '-max_seq_len', 'NONE',
#                                      '-mask_cls_sep',
#                                      '-gpu'])

args = get_args_parser().parse_args(['-model_dir', 'D:\AppCentral\chinese_L-12_H-768_A-12',
                                     '-max_batch_size', '10',
                                     '-max_seq_len', '20',
                                     '-num_worker', '8'
                                    ])

def cos_similar(sen_a_vec, sen_b_vec):
    '''
    计算两个句子的余弦相似度
    '''
    vector_a = np.mat(sen_a_vec)
    vector_b = np.mat(sen_b_vec)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    return cos



def read_corpus(file):
    with open(file,'r',encoding='utf8',errors='ignore') as f:
        list = []
        lines = f.readlines()
        for i in lines:
            list.append(i)
    return list

questions = read_corpus('./Q.txt')
answers = read_corpus('./A.txt')

def main():
    # for i in questions:
    #     print(i)
    bc = BertClient()
    newQuestion = input("请输入您的问题：")
    readyAnswer = "对不起，找不到您的答案"
    iniIndex = 0
    iniSim=0.9
    find = False
    for i in questions:
        doc_vecs = bc.encode([newQuestion, i])
        similarity = cos_similar(doc_vecs[0], doc_vecs[1])
        print(similarity)
    #     if(similarity>iniSim):
    #         iniIndex = questions.index(i)
    #         iniSim = similarity
    #         find = True
    # if find==True:
    #     print(answers[iniIndex])
    #     # print(iniSim)
    # else:
    #     print(readyAnswer)


    # print(doc_vecs)
    # doc_vecs1 = bc.encode(['图书馆的活动可以在哪里找到？', '哪里可以使用图书馆电脑？'])

    # similarity1 = cos_similar(doc_vecs1[0], doc_vecs1[1])
    # print(similarity1)
    # print(questions[2])

    # if(similarity>iniSim):
    #     print('奇怪')
    # print(questions)
def find(sentence):
    bc = BertClient()
    iniIndex = 0
    iniSim=0.92
    find = False
    for i in questions:
        doc_vecs = bc.encode([sentence, i])
        similarity = cos_similar(doc_vecs[0], doc_vecs[1])
        # print(similarity)
        if(similarity>=iniSim):
            iniIndex = questions.index(i)
            iniSim = similarity
            find = True
    if find==True:
        # return iniSim
        return answers[iniIndex]

        # print(answers[iniIndex])
        # print(iniSim)
    else:
        return ''

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
    # vocab_model = pickle.load(open("./models/vocab.pkl", "rb"))
    pass
    # server = BertServer(args)
    # server.start()

    # a = find('我好难过')
    # print(a)

    # main()