# # _*_ coding: utf-8 _*_
# import web
# import os
# import hashlib
# import time
# # from lxml import etree
# from talk import talk
# from image import img
# from build_vocab import Vocab
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
# class WeixinInterface:
#     def __init__(self):
#         self.app_root = os.path.dirname(__file__)
#         self.templates_root = os.path.join(self.app_root, 'templates')
#         self.render = web.template.render(self.templates_root)
#
#     def GET(self):
#         data = web.input()
#         signature = data.signature
#         timestamp = data.timestamp
#         nonce = data.nonce
#         echostr = data.echostr
#         token = "wxpython"
#
#         l = [token, timestamp, nonce]
#         l.sort()
#         sha1 = hashlib.sha1()
#         sha1.update(l[0].encode('utf-8'))
#         sha1.update(l[1].encode('utf-8'))
#         sha1.update(l[2].encode('utf-8'))
#         hashcode = sha1.hexdigest()
#         #python3与python2的加密算法不同！
#         # l = [token, timestamp, nonce]
#         # l.sort()
#         # sha1 = hashlib.sha1()
#         # map(sha1.update, 1)
#         # hashcode = sha1.hexdigest()
#
#         if hashcode == signature:
#             return echostr
#
#     def POST(self):
#         str_xml = web.data()
#         xml = etree.fromstring(str_xml)
#         msgType = xml.find("MsgType").text
#         fromUser = xml.find("FromUserName").text
#         userid = fromUser[0:15]
#         toUser = xml.find("ToUserName").text
#
#
#
#         if  msgType == 'text':
#             content = xml.find("Content").text
#             text = talk(content, userid)
#             return self.render.reply_text(fromUser, toUser, int(time.time()), text)
#         elif msgType == "voice":
#             content = xml.find("Recognition").text
#             text = talk(content, userid)
#             return self.render.reply_text(fromUser, toUser, int(time.time()), text)
#         elif msgType == 'image':
#             content = xml.find("PicUrl").text
#             # data = img(content)
#             return self.render.reply_text(fromUser, toUser, int(time.time()), content)
#         elif msgType == 'event':
#             event = xml.find("Event").text
#             if event == 'subscribe':
#                 return self.render.reply_text(fromUser, toUser, int(time.time()), u'欢迎关注十九春~~' + '\n'
#                                               + u'我可以陪你聊天哦，讲个笑话或者故事，还可以查询快递、日历及车次信息哦。' +
#                                               u'如果你喜欢做饭，我还可以提供菜谱呢！')
#
#         else:
#             return ''
#
