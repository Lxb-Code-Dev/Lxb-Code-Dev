import json
from datetime import datetime
from GetAnswer2 import get_answer
from flask import Flask,request
from flask_cors import CORS, cross_origin
from build_vocab import Vocab
from talk import talk
from imdb_lstm_model import sentiment_ana
app = Flask(__name__)
flag = True
# CORS(app, supports_credentials=True)
@app.route("/chat",methods=['GET','POST'])
@cross_origin(supports_credentials=True)
def upfile():
    '''获取文件'''
    return json.dumps({"a":"b"})
    hour = datetime.now().hour
    global flag
    if (hour == 3 and flag):
        # 调用一次该函数
        flag = False
        attention = sentiment_ana()
        # 判断attention是否为空，后续考虑不为空的情形
    if (hour == 4):
        flag = True
    today = datetime.now().day
    data = request.get_data()
    data = json.loads(data)
    question = data['question']
    userName = data['userName']
    res, score = get_answer(question)
    filepath = "message_log" + "/" + userName + ".txt"
    # a表示已追加写方式打开文件
    f = open(filepath, "a",encoding='utf-8')
    # 写入
    f.writelines(str(today) + ',' + str(question + '\n'))
    f.close()
    if score < 0.3:
        reply = res
    else:
        reply = talk(question)
    # reply = talk(question)
    re = {"code": 200, "reply":reply, "emotion": "正向"}
    return json.dumps(re)
if __name__=='__main__':
    app.run(host="0.0.0.0")

