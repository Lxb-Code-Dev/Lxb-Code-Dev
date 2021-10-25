


import socket,binascii,time

sendsocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
ip = "192.168.1.3"
port = 102

sendsocket.connect((ip, port))
sendsocket.send(binascii.unhexlify('0300001611e00000000100c1020101c2020101c0010a'))
time.sleep(0.1)
sendsocket.send(binascii.unhexlify('0300001902f08032010000ccc100080000f0000001000103c0'))
time.sleep(0.1)
sendsocket.send(binascii.unhexlify('0300002502f0803201000000a30014000028000000000000fd000009505f50524f4752414d0300002102f0803201000000330010000029000000000009505f50524f4752414d'))
# # time.sleep(0.1)
# sendsocket.send(binascii.unhexlify('0300002102f0803201000000330010000029000000000009505f50524f4752414d'))
# time.sleep(0.1)
#sendsocket.send(binascii.unhexlify('0300002502f0803201000000a30014000028000000000000fd000009505f50524f4752414d'))
time.sleep(0.1)
sendsocket.close()


