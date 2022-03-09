# from datetime import datetime
# import falcon
# from wechatpy.utils import check_signature
# from wechatpy.exceptions import InvalidSignatureException
# from wechatpy import parse_message
# from wechatpy.replies import TextReply, ImageReply
# from GetAnswer2 import get_answer
# from talk import talk
# from prepro import *
#
# import pickle
# # from build_vocab import Vocab
# #该全局变量用于情感分析
# import imdb_lstm_model
#
# flag = True
#
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
# class Connect(object):
#
#     def on_get(self, req, resp):
#         query_string = req.query_string
#         query_list = query_string.split('&')
#         b = {}
#         for i in query_list:
#             b[i.split('=')[0]] = i.split('=')[1]
#
#         try:
#             check_signature(token='wxpython', signature=b['signature'], timestamp=b['timestamp'], nonce=b['nonce'])
#             resp.body = (b['echostr'])
#         except InvalidSignatureException:
#             pass
#         resp.status = falcon.HTTP_200
#
#     def on_post(self, req, resp):
#         xml = req.stream.read()
#         msg = parse_message(xml)
#         #对二进制xml进行解码，得到xml语言，然后将其中的换行符号去掉，便于下面查找字段
#         myxml = xml.decode().replace('\n', '')
#         #查找FromUserName字段，得到openid
#         usr_open_id = myxml[myxml.find("<FromUserName><![CDATA[") + len("<FromUserName><![CDATA["):myxml.find(
#             "]]></FromUserName>")]
#         #日期
#         today = datetime.now().day
#         if msg.type == 'text':
#             #问答匹配，在答案库中检索
#             res, score = get_answer(msg.content)
#             # res = demo.find(msg.content)
#             #相似度大于2时返回
#             if score<0.3:
#             # if res!='':
#                 myreply = res
#             else:
#                 myreply = talk(msg.content)
#             reply = TextReply(content=myreply, message=msg)
#             xml = reply.render()
#             resp.body = (xml)
#             resp.status = falcon.HTTP_200
#             #消息记录保存路径
#             filepath = "message_log" + "/" + str(usr_open_id) + ".txt"
#             #a表示已追加写方式打开文件
#             f=open(filepath,"a")
#             #写入
#             f.writelines(str(today)+','+str(msg.content)+'\n')
#             f.close()
#             hour = datetime.now().hour
#             global flag
#             if(hour==3 and flag):
#                 #调用一次该函数
#                 flag = False
#                 attention = imdb_lstm_model.sentiment_ana()
#                 #判断attention是否为空，后续考虑不为空的情形
#             if(hour==4):
#                 flag = True
#         if msg.type=='voice':
#             reply = TextReply(content=msg.recognition, message=msg)
#             xml = reply.render()
#             resp.body = (xml)
#             resp.status = falcon.HTTP_200
#             #消息记录保存路径
#             filepath = "message_log" + "/" + str(usr_open_id) + ".txt"
#             #a表示已追加写方式打开文件
#             f=open(filepath,"a")
#             #写入
#             f.writelines(str(today)+','+str(msg.recognition)+'\n')
#             f.close()
#             hour = datetime.now().hour
#             if(hour==3 and flag):
#                 #调用一次该函数
#                 flag = False
#                 attention = imdb_lstm_model.sentiment_ana()
#                 #判断attention是否为空，后续考虑不为空的情形
#             if(hour==4):
#                 flag = True
#
#         else:
#             pass
# # if __name__ == '__main__':
# #     print(datetime.now().day)
# # voc_model = pickle.load(open("./models/vocab.pkl", "rb"))
# app = falcon.API()
# connect = Connect()
# app.add_route('/connect', connect)