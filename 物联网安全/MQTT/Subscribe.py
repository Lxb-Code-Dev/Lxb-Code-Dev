import base64

import paho.mqtt.client as mqtt
import time
from Crypto import Random
from Crypto.Hash import SHA
from Crypto.Cipher import PKCS1_v1_5 as Cipher_pkcs1_v1_5
from Crypto.Signature import PKCS1_v1_5 as Signature_pkcs1_v1_5
from Crypto.PublicKey import RSA
class subscribe:
    def __init__(self):
        self.name=''
        self.state=0
        self.isfree=1
        self.username=[]
    def on_message(self, client, userdata, msg):
        print(msg.topic + " " + str(msg.payload))
        with open("client-private.pem") as f:
            key = f.read()
            rsakey = RSA.importKey(key)
            cipher = Cipher_pkcs1_v1_5.new(rsakey)
            random_generator = Random.new().read
            text = cipher.decrypt(base64.b64decode(msg.payload), random_generator)
        content=text.decode('utf-8')
        print(content)
        if time.time()-float(content.split(' ')[1])>5:
            return
        if self.isfree==0:
            if content.find('_')!=-1:
                #控制单一设备
                if len(content.split('_'))==2 and (content.split('_')[0].split('/')[0] in self.username) and (content.split('_')[0].split('/')[1]==self.name):
                    if content.split('_')[1].split(' ')[0]=="check":
                        sendbuf=self.name+"处于关闭状态" if self.state == 0 else self.name+"处于开启状态"
                        with open("server-public.pem") as f:
                            key = f.read()
                            rsakey = RSA.importKey(key)
                            cipher = Cipher_pkcs1_v1_5.new(rsakey)
                            cipher_text = base64.b64encode(cipher.encrypt(sendbuf.encode('utf-8')))
                        client.publish(self.name, payload=cipher_text, qos=0)
                    elif content.split('_')[1].split(' ')[0] == "off":
                        self.state = 0
                        sendbuf = self.name + "已关闭"
                        with open("server-public.pem") as f:
                            key = f.read()
                            rsakey = RSA.importKey(key)
                            cipher = Cipher_pkcs1_v1_5.new(rsakey)
                            cipher_text = base64.b64encode(cipher.encrypt(sendbuf.encode('utf-8')))
                        client.publish(self.name, payload=cipher_text, qos=0)

                    elif content.split('_')[1].split(' ')[0] == "on":
                        self.state = 1
                        sendbuf = self.name + "已打开"
                        with open("server-public.pem") as f:
                            key = f.read()
                            rsakey = RSA.importKey(key)
                            cipher = Cipher_pkcs1_v1_5.new(rsakey)
                            cipher_text = base64.b64encode(cipher.encrypt(sendbuf.encode('utf-8')))
                        client.publish(self.name, payload=cipher_text, qos=0)

                    elif content.split('_')[1].split(' ')[0].split('to')[0]=="share":
                        self.username.append(content.split('_')[1].split('to')[1])
                        sendbuf = self.name + "设备转发绑定成功"
                        with open("server-public.pem") as f:
                            key = f.read()
                            rsakey = RSA.importKey(key)
                            cipher = Cipher_pkcs1_v1_5.new(rsakey)
                            cipher_text = base64.b64encode(cipher.encrypt(sendbuf.encode('utf-8')))
                        client.publish(self.name, payload=cipher_text, qos=0)

                    else:
                        sendbuf = "指令错误"
                        with open("server-public.pem") as f:
                            key = f.read()
                            rsakey = RSA.importKey(key)
                            cipher = Cipher_pkcs1_v1_5.new(rsakey)
                            cipher_text = base64.b64encode(cipher.encrypt(sendbuf.encode('utf-8')))
                        client.publish(self.name, payload=cipher_text, qos=0)

        if self.isfree==1 and len(content.split('_'))==2 and content.split('_')[1].split(' ')[0]=='link':
            self.username.append(content.split('/')[0])
            self.isfree=0
            sendbuf = "link"
            with open("server-public.pem") as f:
                key = f.read()
                rsakey = RSA.importKey(key)
                cipher = Cipher_pkcs1_v1_5.new(rsakey)
                cipher_text = base64.b64encode(cipher.encrypt(sendbuf.encode('utf-8')))
            client.publish(self.name, payload=cipher_text, qos=0)


    def run(self):
        client=mqtt.Client()
        self.name=input('设备名称：')
        client.on_message=self.on_message
        client.connect('127.0.0.1',1883,600)
        topic=input('Enter the topic you want to subscribe: ')
        client.subscribe(topic,qos=0)
        client.loop_forever()

subscribe1=subscribe()
subscribe1.run()