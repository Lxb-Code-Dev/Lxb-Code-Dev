import paho.mqtt.client as mqtt
def on_message(client,userdata,msg):
    print()
client=mqtt.Client()
client.on_message=on_message
client.connect('127.0.0.1',1883,600)

client.loop_start()

while True:
    topic=input('Enter the topic name: ')
    message=input('Enter the message to send: ')
    client.publish(topic,payload=message,qos=0)