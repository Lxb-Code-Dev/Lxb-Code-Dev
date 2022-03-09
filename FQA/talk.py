# _*_ coding: utf-8 _*_
import requests
import json
from build_vocab import Vocab
import pickle
def talk(content):
    url = 'http://www.tuling123.com/openapi/api'
    s = requests.session()
    d = {"key":"dd814a763ef54870ae22adc501444be5", "info": content}
    data = json.dumps(d)
    r = s.post(url, data=data)
    text = json.loads(r.text)
    code = text["code"]
    if code ==100000:
        result = text["text"]
    elif code == 200000:
        result = text["text"] + '\n' + text["url"]
    elif code == 302000:
        result = text["text"] + '\n' + text["list"][0]["article"]
    elif code == 308000:
        result = text["text"] + '\n' + text["list"][0]["info"] + text["list"][0]["detailurl"]
    return result

# voc_model = pickle.load(open("./models/vocab.pkl", "rb"))


