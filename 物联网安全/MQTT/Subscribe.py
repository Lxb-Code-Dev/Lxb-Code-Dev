import paho.mqtt.client as mqtt

class subscribe:
    def __init__(self):
        self.name=''
        self.state=0
        self.isfree=1
        self.username=''
    def on_message(self,client,userdata,msg):
        print(msg.topic + " " + str(msg.payload))
        content=str(msg.payload, 'utf-8')
        if self.isfree==0:
            if content.find('_')==-1 and len(content.split('/'))==2 and content.split('/')[0]==self.username :
                #控制所有设备
                if content.split('/')[1]=="check":
                    client.publish(self.name, payload="关闭" if self.state==0 else "开启", qos=0)
                elif content.split('/')[1]=="off":
                    self.state=0
                    client.publish(self.name, payload=self.name+"已关闭", qos=0)
                elif content.split('/')[1]=="on" :
                    self.state = 1
                    client.publish(self.name, payload=self.name + "已打开", qos=0)
                else:
                    client.publish(self.name, payload="指令错误", qos=0)
            if content.find('_')!=-1:
                #控制单一设备
                if len(content.split('_'))==2 and (content.split('_')[0].split('/')[0]==self.username):
                    if content.split('_')[0].split('/')[1]=="check":
                        client.publish(self.name, payload=self.name+"处于关闭状态" if self.state == 0 else self.name+"处于开启状态", qos=0)
                    elif content.split('_')[0].split('/')[1] == "off":
                        self.state = 0
                        client.publish(self.name, payload=self.name + "已关闭", qos=0)
                    elif content.split('_')[0].split('/')[1] == "on":
                        self.state = 1
                        client.publish(self.name, payload=self.name + "已打开", qos=0)
                    else:
                        client.publish(self.name, payload="指令错误", qos=0)
        if self.isfree==1 and len(content.split('_'))==2 and content.split('_')[1]=='link':
            self.username=content.split('/')[0]
            self.isfree=0
            client.publish(self.name, payload="link", qos=0)

    def run(self):
        client=mqtt.Client()
        self.name=input('Your name：')
        client.on_message=self.on_message
        client.connect('127.0.0.1',1883,600)
        topic=input('Enter the topic you want to subscribe: ')
        client.subscribe(topic,qos=0)
        client.loop_forever()

subscribe1=subscribe()
subscribe1.run()