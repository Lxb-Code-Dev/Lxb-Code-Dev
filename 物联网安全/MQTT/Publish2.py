import paho.mqtt.client as mqtt
import tkinter as tk
import time
topics=[]
class publish():
    def __init__(self,master,user,client,topic):
        self.user=user
        self.topic=topic
        self.client=client
        self.root=master
        self.root.title('设备管理')
        self.root.geometry('600x600')
        # 标签
        l1 = tk.Label(self.root, text='设备管理GUI', font=('宋体', 40), width=40, height=3)
        l1.pack()
        tk.Label(self.root, text='Name:', font=('宋体', 17)).place(x=180, y=230)
        tk.Label(self.root, text='Message:', font=('宋体', 17)).place(x=170, y=300)
        # 用户名输入框
        self.usrname = tk.StringVar()
        self.entry_usr_name = tk.Entry(self.root, textvariable=self.usrname)
        self.entry_usr_name.place(x=270, y=235)
        # 密码输入框
        self.usrpwd = tk.StringVar()
        self.entry_usr_pwd = tk.Entry(self.root, textvariable=self.usrpwd)
        self.entry_usr_pwd.place(x=270, y=305)
        bt_login = tk.Button(self.root, text='发送', command=self.send)
        bt_login.place(x=240, y=400)
        self.root.mainloop()
    def send(self):
        name = self.usrname.get()
        message = self.usrpwd.get()
        client.publish(topic, payload=self.user + '/' +name+'_'+ message+' '+str(0), qos=0)
        if message == 'link':
            client.subscribe(name, qos=0)

#str(time.time())

def on_message(client,userdata,msg):
    print(msg.topic + " " + str(msg.payload.decode()))
client=mqtt.Client()
client.on_message=on_message
client.connect('127.0.0.1',1883,600)
client.loop_start()
username=input('Enter user name: ')
topic=input('Enter the topic name: ')

gui=tk.Tk()
publish(gui,username,client,topic)