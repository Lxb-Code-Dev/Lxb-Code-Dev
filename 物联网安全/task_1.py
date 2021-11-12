


import socket,binascii,time

sendsocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
ip = "127.0.0.1"
port = 1883

sendsocket.connect((ip, port))
sendsocket.send(binascii.unhexlify('100c00044d515454040202580000'))
time.sleep(1)
sendsocket.send(binascii.unhexlify('30280007636f6e74726f6c6c78622f6c696768745f6f6e20313633363630363030312e31353735393636'))

sendsocket.close()


