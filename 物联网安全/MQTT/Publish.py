import paho.mqtt.client as mqtt
topics=[]
def on_message(client,userdata,msg):
    print(msg.topic + " " + str(msg.payload))


client=mqtt.Client()
client.on_message=on_message
client.connect('127.0.0.1',1883,600)
client.loop_start()
username=input('Enter user name: ')
topic=input('Enter the topic name: ')
while True:
    message=input('Enter the message to send: ')
    client.publish(topic,payload=username+'/'+message,qos=0)
    if message.split('_')[1]=='link':
        client.subscribe(message.split('_')[0], qos=0)