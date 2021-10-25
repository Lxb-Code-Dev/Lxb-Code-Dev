#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#include<WinSock2.h>
#include<Ws2tcpip.h>
#include<cstring>
#include<string>
#include <map>
#include<time.h>
#include<set>//实现群聊
#pragma comment(lib, "ws2_32.lib")   //加载 ws2_32.dll到此项目
using namespace std;


//定义缓冲区
const int BUF_SIZE = 1300;
const int nickname_size = 100;
const int content_size = 1000;
const int time_size = 100;
const int MAX_CLIENT_NUM = 20;
int group_num = 0;
int NOW_CLIENT_NUM = 0;
int nSize = sizeof(SOCKADDR);
char SendBuf[BUF_SIZE];
char RecBuf[BUF_SIZE];
char InputBuf[BUF_SIZE];
string from_nickname;
string to_nickname;
string time_stamp;
string my_content;
string SendMsg;
time_t timep;
//每个客户端安排一个线程处理
HANDLE hListen_Thread[MAX_CLIENT_NUM];
HANDLE hRecv_Thread[MAX_CLIENT_NUM];
map<string, SOCKET> name_soc;
//群聊结构体<群名，成员集合>
map<int, set<SOCKET>> group_set_map;
//已连接的客户端的信息结构体
struct CLIENT
{
    SOCKET sock;
    SOCKADDR_IN CliAddr;
    char nickname[100] = "";
}Cli[MAX_CLIENT_NUM];

//消息结构体
struct msg
{
    string from_nickname = "";
    string to_nickname = "";
    string timestamp = "";
    string send_content = "";
};




//声明字符串填充函数，对发送消息进行分段，每个段有固定格式，以#结尾，剩余长度填充为空格
string msg_pad(string msg, int length)
{
    int round = length - msg.length();
    msg += '#';
    for (int i = 1; i < round; i++) {
        msg += " ";
    }
    return msg;
}
//对收到的信息按照传输协议进行拆分
msg split_msg(char* a)
{

    msg transpond_msg;
    string s_msg = string(a);

    string temp = s_msg.substr(0, 100);
    int x = temp.find_last_of("#");
    transpond_msg.from_nickname = temp.substr(0, x);

    temp = s_msg.substr(100, 100);
    x = temp.find_last_of("#");
    transpond_msg.to_nickname = temp.substr(0, x);

    temp = s_msg.substr(200, 100);
    x = temp.find_last_of("#");
    transpond_msg.timestamp = temp.substr(0, x - 1);

    temp = s_msg.substr(300, 1000);
    x = temp.find_last_of("#");
    transpond_msg.send_content = temp.substr(0, x);
    return transpond_msg;
}
//接收消息请求线程
DWORD WINAPI Server_Rec_Thread(LPVOID lp)
{
    int i = (int)lp;

    while (Cli[i].sock != INVALID_SOCKET)
    {
        memset(RecBuf, '\0', sizeof(RecBuf));
        int len_recv = recv(Cli[i].sock, RecBuf, BUF_SIZE, 0);
        if (len_recv > 0)
        {
            msg msg_rec = split_msg(RecBuf);
            //初次登录登记请求
            if (strcmp(msg_rec.to_nickname.c_str(), "serve") == 0 && strcmp(msg_rec.send_content.c_str(), "link_serve") == 0)
            {

                map<string, SOCKET>::iterator it = name_soc.find(msg_rec.from_nickname);
                if (it != name_soc.end() || strcmp(msg_rec.from_nickname.c_str(), "serve") == 0)
                {
                    cout << "用户登录失败，该用户名【" + msg_rec.from_nickname + "】已被占用。" << "     <" << msg_rec.timestamp << ">" << endl;
                    from_nickname = "serve";
                    from_nickname = msg_pad(from_nickname, nickname_size);
                    to_nickname = msg_rec.from_nickname;
                    to_nickname = msg_pad(to_nickname, nickname_size);
                    time(&timep);
                    time_stamp = ctime(&timep);
                    time_stamp = msg_pad(time_stamp, time_size);
                    my_content = "rename";
                    my_content = msg_pad(my_content, content_size);
                    SendMsg = from_nickname + to_nickname + time_stamp + my_content;
                    memset(SendBuf, '\0', sizeof(SendBuf));
                    strcpy(SendBuf, SendMsg.c_str());
                    send(Cli[i].sock, SendBuf, sizeof(SendBuf), NULL);
                    continue;
                }
                //确认注册
                from_nickname = "serve";
                from_nickname = msg_pad(from_nickname, nickname_size);
                to_nickname = msg_rec.from_nickname;
                to_nickname = msg_pad(to_nickname, nickname_size);
                time(&timep);
                time_stamp = ctime(&timep);
                time_stamp = msg_pad(time_stamp, time_size);
                my_content = "confirm";
                my_content = msg_pad(my_content, content_size);
                SendMsg = from_nickname + to_nickname + time_stamp + my_content;
                memset(SendBuf, '\0', sizeof(SendBuf));
                strcpy(SendBuf, SendMsg.c_str());
                send(Cli[i].sock, SendBuf, sizeof(SendBuf), NULL);
                name_soc.insert(pair<string, SOCKET>(msg_rec.from_nickname, Cli[i].sock));
                cout << "【" << msg_rec.from_nickname << "】" << "上线了！" << "     <" << msg_rec.timestamp << ">" << endl;
                from_nickname = "serve";
                from_nickname = msg_pad(from_nickname, nickname_size);
                to_nickname = string(Cli[i].nickname);
                to_nickname = msg_pad(to_nickname, nickname_size);
                time(&timep);
                time_stamp = ctime(&timep);
                time_stamp = msg_pad(time_stamp, time_size);
                my_content = "创建连接成功，您已成功登录到聊天系统。";
                my_content = msg_pad(my_content, content_size);
                SendMsg = from_nickname + to_nickname + time_stamp + my_content;
                memset(SendBuf, '\0', sizeof(SendBuf));
                strcpy(SendBuf, SendMsg.c_str());
                send(Cli[i].sock, SendBuf, sizeof(SendBuf), NULL);
                from_nickname = "serve";
                from_nickname = msg_pad(from_nickname, nickname_size);
                time(&timep);
                time_stamp = ctime(&timep);
                time_stamp = msg_pad(time_stamp, time_size);
                my_content = "用户【" + msg_rec.from_nickname + "】上线了。";
                my_content = msg_pad(my_content, content_size);
                //通知其他用户，有人上线了
                map<string, SOCKET>::iterator it1 = name_soc.begin();
                for (it1; it1 != name_soc.end(); it1++)
                {
                    if (it1->first != msg_rec.from_nickname && it1->second != INVALID_SOCKET)
                    {
                        to_nickname = it1->first;
                        to_nickname = msg_pad(to_nickname, nickname_size);
                        SendMsg = from_nickname + to_nickname + time_stamp + my_content;
                        memset(SendBuf, '\0', sizeof(SendBuf));
                        strcpy(SendBuf, SendMsg.c_str());
                        send(it1->second, SendBuf, sizeof(SendBuf), NULL);
                    }

                }


                continue;
            }
            //申请创建群聊
            if (strcmp(msg_rec.to_nickname.c_str(), "serve") == 0 && strcmp(msg_rec.send_content.c_str(), "create_group") == 0)
            {
                set<SOCKET> sok_set;
                sok_set.insert(Cli[i].sock);
                group_set_map.insert(pair<int, set<SOCKET>>(group_num, sok_set));
                cout<<"【"<<msg_rec.from_nickname<<"】创建群聊，服务器分配群号为"<< group_num<< "     <" << msg_rec.timestamp << ">" << endl;
                from_nickname = "serve";
                from_nickname = msg_pad(from_nickname, nickname_size);
                to_nickname = string(Cli[i].nickname);
                to_nickname = msg_pad(to_nickname, nickname_size);
                time(&timep);
                time_stamp = ctime(&timep);
                time_stamp = msg_pad(time_stamp, time_size);
                my_content = "群聊创建成功，群号为"+to_string(group_num);
                my_content = msg_pad(my_content, content_size);
                SendMsg = from_nickname + to_nickname + time_stamp + my_content;
                memset(SendBuf, '\0', sizeof(SendBuf));
                strcpy(SendBuf, SendMsg.c_str());
                send(Cli[i].sock, SendBuf, sizeof(SendBuf), NULL);
                group_num++;
                continue;
            }
            //在群里发消息，进入群聊模式
            if (strcmp(msg_rec.to_nickname.c_str(), "serve") == 0 && strcmp(msg_rec.send_content.c_str(), "group_chat") == 0)
            {
                cout<<"【"<< msg_rec.from_nickname<<"】进入群聊模式=========="<< "     <" << msg_rec.timestamp << ">" << endl;
                from_nickname = "serve";
                from_nickname = msg_pad(from_nickname, nickname_size);
                to_nickname = string(Cli[i].nickname);
                to_nickname = msg_pad(to_nickname, nickname_size);
                time(&timep);
                time_stamp = ctime(&timep);
                time_stamp = msg_pad(time_stamp, time_size);
                my_content = "您已进入群聊模式==========" ;
                my_content = msg_pad(my_content, content_size);
                SendMsg = from_nickname + to_nickname + time_stamp + my_content;
                memset(SendBuf, '\0', sizeof(SendBuf));
                strcpy(SendBuf, SendMsg.c_str());
                send(Cli[i].sock, SendBuf, sizeof(SendBuf), NULL);
                int offline = 0;
                while (1)
                {
                    memset(RecBuf, '\0', sizeof(RecBuf));
                    int len_recv_group = recv(Cli[i].sock, RecBuf, BUF_SIZE, 0);
                    if (len_recv_group > 0)
                    {
                        //群聊消息
                        msg grmsg_rec = split_msg(RecBuf);
                        //退出程序
                        if (strcmp(grmsg_rec.to_nickname.c_str(), "serve") == 0 && strcmp(grmsg_rec.send_content.c_str(), "quit") == 0)
                        {
                            closesocket(Cli[i].sock);
                            cout << "【" << grmsg_rec.from_nickname << "】" << "下线了" << "     <" << grmsg_rec.timestamp << ">" << endl;
                            from_nickname = "serve";
                            from_nickname = msg_pad(from_nickname, nickname_size);
                            time(&timep);
                            time_stamp = ctime(&timep);
                            time_stamp = msg_pad(time_stamp, time_size);
                            my_content = "用户【" + grmsg_rec.from_nickname + "】下线了。";
                            my_content = msg_pad(my_content, content_size);
                            //通知其他用户，有人下线了
                            map<string, SOCKET>::iterator it = name_soc.begin();
                            string del_client;
                            int flag = 0;
                            for (it; it != name_soc.end(); it++)
                            {
                                if (it->first != msg_rec.from_nickname)
                                {
                                    to_nickname = it->first;
                                    to_nickname = msg_pad(to_nickname, nickname_size);
                                    SendMsg = from_nickname + to_nickname + time_stamp + my_content;
                                    memset(SendBuf, '\0', sizeof(SendBuf));
                                    strcpy(SendBuf, SendMsg.c_str());
                                    send(it->second, SendBuf, sizeof(SendBuf), NULL);
                                }
                                else
                                {
                                    del_client = it->first;
                                    flag = 1;
                                }
                            }
                            if (flag)
                                name_soc.erase(del_client);
                            offline = 1;
                            break;
                        }
                        //退出群聊
                        if (strcmp(grmsg_rec.to_nickname.c_str(), "serve") == 0 && strcmp(grmsg_rec.send_content.c_str(), "quit_group") == 0)
                        {
                            cout << "【" << grmsg_rec.from_nickname << "】退出群聊模式==========" << "     <" << grmsg_rec.timestamp << ">" << endl;
                            from_nickname = "serve";
                            from_nickname = msg_pad(from_nickname, nickname_size);
                            to_nickname = string(Cli[i].nickname);
                            to_nickname = msg_pad(to_nickname, nickname_size);
                            time(&timep);
                            time_stamp = ctime(&timep);
                            time_stamp = msg_pad(time_stamp, time_size);
                            my_content = "您已退出群聊模式==========";
                            my_content = msg_pad(my_content, content_size);
                            SendMsg = from_nickname + to_nickname + time_stamp + my_content;
                            memset(SendBuf, '\0', sizeof(SendBuf));
                            strcpy(SendBuf, SendMsg.c_str());
                            send(Cli[i].sock, SendBuf, sizeof(SendBuf), NULL);
                            break;
                        }
                        if (to_string(atoi(grmsg_rec.to_nickname.c_str())) != grmsg_rec.to_nickname)
                        {
                            cout << "没有找到群聊【" << grmsg_rec.to_nickname << "】" << "     <" << msg_rec.timestamp << ">" << endl;
                            from_nickname = "serve";
                            from_nickname = msg_pad(from_nickname, nickname_size);
                            to_nickname = grmsg_rec.from_nickname;
                            to_nickname = msg_pad(to_nickname, nickname_size);
                            time(&timep);
                            time_stamp = ctime(&timep);
                            time_stamp = msg_pad(time_stamp, time_size);
                            my_content = "抱歉，没有对应群聊，请检查输入是否正确。";
                            my_content = msg_pad(my_content, content_size);
                            SendMsg = from_nickname + to_nickname + time_stamp + my_content;
                            memset(SendBuf, '\0', sizeof(SendBuf));
                            strcpy(SendBuf, SendMsg.c_str());
                            send(Cli[i].sock, SendBuf, sizeof(SendBuf), NULL);
                            continue;
                        }
                        map<int,set<SOCKET>>::iterator it4 = group_set_map.find(atoi(grmsg_rec.to_nickname.c_str()));
                        if (it4 == group_set_map.end())
                        {
                            cout << "没有找到群聊【" << grmsg_rec.to_nickname << "】" << "     <" << msg_rec.timestamp << ">" << endl;
                            from_nickname = "serve";
                            from_nickname = msg_pad(from_nickname, nickname_size);
                            to_nickname = grmsg_rec.from_nickname;
                            to_nickname = msg_pad(to_nickname, nickname_size);
                            time(&timep);
                            time_stamp = ctime(&timep);
                            time_stamp = msg_pad(time_stamp, time_size);
                            my_content = "抱歉，没有对应群聊，请检查输入是否正确。";
                            my_content = msg_pad(my_content, content_size);
                            SendMsg = from_nickname + to_nickname + time_stamp + my_content;
                            memset(SendBuf, '\0', sizeof(SendBuf));
                            strcpy(SendBuf, SendMsg.c_str());
                            send(Cli[i].sock, SendBuf, sizeof(SendBuf), NULL);
                            continue;
                        }
                        else
                        {
                            if (it4->second.count(Cli[i].sock) == 0)
                            {
                                from_nickname = "serve";
                                from_nickname = msg_pad(from_nickname, nickname_size);
                                to_nickname = grmsg_rec.from_nickname;
                                to_nickname = msg_pad(to_nickname, nickname_size);
                                time(&timep);
                                time_stamp = ctime(&timep);
                                time_stamp = msg_pad(time_stamp, time_size);
                                my_content = "抱歉，您未加入该群聊，发送失败。";
                                my_content = msg_pad(my_content, content_size);
                                SendMsg = from_nickname + to_nickname + time_stamp + my_content;
                                memset(SendBuf, '\0', sizeof(SendBuf));
                                strcpy(SendBuf, SendMsg.c_str());
                                send(Cli[i].sock, SendBuf, sizeof(SendBuf), NULL);
                                cout << "用户【" << grmsg_rec.from_nickname << "】未加入群聊【" << grmsg_rec.to_nickname << "】，发送信息失败。" << "     <" << msg_rec.timestamp << ">" << endl;
                                continue;
                            }
                            else
                            {
                                set<SOCKET>::iterator it5 = it4->second.begin();
                                for (it5; it5 != it4->second.end(); it5++)
                                {
                                    if(*it5!=INVALID_SOCKET)
                                        send(*it5, RecBuf, BUF_SIZE, NULL);
                                }
                                cout << "【" << grmsg_rec.from_nickname << "】在群聊【"<< grmsg_rec.to_nickname<<"】中发送：" <<grmsg_rec.send_content<< "     <" << msg_rec.timestamp << ">" << endl;
                                continue;
                            }
                        }
                        
                    }
                }
                if (offline)
                    break;
                continue;
                
            }
            //加入群聊
            if (strcmp(msg_rec.to_nickname.c_str(), "serve") == 0 && strcmp(msg_rec.send_content.substr(0,10).c_str(), "join_group") == 0)
            {
                int group_n = atoi(msg_rec.send_content.substr(10, msg_rec.send_content.length()).c_str());
                if (to_string(group_n) != msg_rec.send_content.substr(10, msg_rec.send_content.length()))
                {
                    cout << "未找到群号为" << msg_rec.send_content.substr(10, msg_rec.send_content.length()) << "的群聊。" << "     <" << msg_rec.timestamp << ">" << endl;
                    from_nickname = "serve";
                    from_nickname = msg_pad(from_nickname, nickname_size);
                    to_nickname = string(Cli[i].nickname);
                    to_nickname = msg_pad(to_nickname, nickname_size);
                    time(&timep);
                    time_stamp = ctime(&timep);
                    time_stamp = msg_pad(time_stamp, time_size);
                    my_content = "加入群聊失败，未找到群聊。";
                    my_content = msg_pad(my_content, content_size);
                    SendMsg = from_nickname + to_nickname + time_stamp + my_content;
                    memset(SendBuf, '\0', sizeof(SendBuf));
                    strcpy(SendBuf, SendMsg.c_str());
                    send(Cli[i].sock, SendBuf, sizeof(SendBuf), NULL);
                    continue;
                }
                map<int, set<SOCKET>>::iterator it = group_set_map.find(group_n);
                if (it == group_set_map.end())
                {
                    cout<<"未找到群号为"<< msg_rec.send_content.substr(10, msg_rec.send_content.length())<<"的群聊。"<< "     <" << msg_rec.timestamp << ">" << endl;
                    from_nickname = "serve";
                    from_nickname = msg_pad(from_nickname, nickname_size);
                    to_nickname = string(Cli[i].nickname);
                    to_nickname = msg_pad(to_nickname, nickname_size);
                    time(&timep);
                    time_stamp = ctime(&timep);
                    time_stamp = msg_pad(time_stamp, time_size);
                    my_content = "加入群聊失败，未找到群聊。" ;
                    my_content = msg_pad(my_content, content_size);
                    SendMsg = from_nickname + to_nickname + time_stamp + my_content;
                    memset(SendBuf, '\0', sizeof(SendBuf));
                    strcpy(SendBuf, SendMsg.c_str());
                    send(Cli[i].sock, SendBuf, sizeof(SendBuf), NULL);
                    continue;
                }
                else
                {
                    it->second.insert(Cli[i].sock);
                    cout << "【"<<msg_rec.from_nickname<<"】加入群聊" << msg_rec.send_content.substr(10, msg_rec.send_content.length()) << "。" << "     <" << msg_rec.timestamp << ">" << endl;
                    from_nickname = "serve";
                    from_nickname = msg_pad(from_nickname, nickname_size);
                    to_nickname = string(Cli[i].nickname);
                    to_nickname = msg_pad(to_nickname, nickname_size);
                    time(&timep);
                    time_stamp = ctime(&timep);
                    time_stamp = msg_pad(time_stamp, time_size);
                    my_content = "【"+ msg_rec.from_nickname +"】加入群聊"+ msg_rec.send_content.substr(10, msg_rec.send_content.length());
                    my_content = msg_pad(my_content, content_size);
                    set<SOCKET>::iterator it1 = it->second.begin();
                    for (it1; it1 != it->second.end(); it1++)
                    {
                        SendMsg = from_nickname + to_nickname + time_stamp + my_content;
                        memset(SendBuf, '\0', sizeof(SendBuf));
                        strcpy(SendBuf, SendMsg.c_str());
                        send(*it1, SendBuf, sizeof(SendBuf), NULL);
                    }
                    continue;

                }
            }
            //下线请求处理
            if (strcmp(msg_rec.to_nickname.c_str(), "serve") == 0 && strcmp(msg_rec.send_content.c_str(), "quit") == 0)
            {
                closesocket(Cli[i].sock);
                cout << "【" << msg_rec.from_nickname << "】" << "下线了" << "     <" << msg_rec.timestamp << ">" << endl;
                from_nickname = "serve";
                from_nickname = msg_pad(from_nickname, nickname_size);
                time(&timep);
                time_stamp = ctime(&timep);
                time_stamp = msg_pad(time_stamp, time_size);
                my_content = "用户【" + msg_rec.from_nickname + "】下线了。";
                my_content = msg_pad(my_content, content_size);
                map<int, set<SOCKET>>::iterator it6 = group_set_map.begin();
                for (it6; it6 != group_set_map.end(); it6++)
                {
                    if (it6->second.count(Cli[i].sock) != 0)
                        it6->second.erase(Cli[i].sock);
                }
                //通知其他用户，有人下线了
                map<string, SOCKET>::iterator it = name_soc.begin();
                string del_client;
                int flag = 0;
                for (it; it != name_soc.end(); it++)
                {
                    if (it->first != msg_rec.from_nickname)
                    {
                        to_nickname = it->first;
                        to_nickname = msg_pad(to_nickname, nickname_size);
                        SendMsg = from_nickname + to_nickname + time_stamp + my_content;
                        memset(SendBuf, '\0', sizeof(SendBuf));
                        strcpy(SendBuf, SendMsg.c_str());
                        send(it->second, SendBuf, sizeof(SendBuf), NULL);
                    }
                    else
                    {
                        del_client = it->first;
                        flag = 1;
                    }
                }
                if (flag)
                    name_soc.erase(del_client);


                break;
            }
            //查询群组处理
            if (strcmp(msg_rec.to_nickname.c_str(), "serve") == 0 && strcmp(msg_rec.send_content.c_str(), "group_list") == 0)
            {
                cout << "【" << msg_rec.from_nickname << "】" << "请求查询群组名单" << "     <" << msg_rec.timestamp << ">" << endl;

                from_nickname = "serve";
                from_nickname = msg_pad(from_nickname, nickname_size);
                to_nickname = msg_rec.from_nickname;
                to_nickname = msg_pad(to_nickname, nickname_size);
                time(&timep);
                time_stamp = ctime(&timep);
                time_stamp = msg_pad(time_stamp, time_size);
                //回复群组列表
                map<int, set<SOCKET>>::iterator it7 = group_set_map.begin();
                for (it7; it7 != group_set_map.end(); it7++)
                {

                    my_content = "【" + to_string(it7->first) + "】";
                    my_content = msg_pad(my_content, content_size);
                    SendMsg = from_nickname + to_nickname + time_stamp + my_content;
                    memset(SendBuf, '\0', sizeof(SendBuf));
                    strcpy(SendBuf, SendMsg.c_str());
                    send(Cli[i].sock, SendBuf, sizeof(SendBuf), NULL);


                }

                continue;
            }
            //查询用户处理
            if (strcmp(msg_rec.to_nickname.c_str(), "serve") == 0 && strcmp(msg_rec.send_content.c_str(), "online_list") == 0)
            {
                cout << "【" << msg_rec.from_nickname << "】" << "请求查询在线名单" << "     <" << msg_rec.timestamp << ">" << endl;

                from_nickname = "serve";
                from_nickname = msg_pad(from_nickname, nickname_size);
                to_nickname = msg_rec.from_nickname;
                to_nickname = msg_pad(to_nickname, nickname_size);
                time(&timep);
                time_stamp = ctime(&timep);
                time_stamp = msg_pad(time_stamp, time_size);
                //回复在线用户列表
                map<string, SOCKET>::iterator it2 = name_soc.begin();
                for (it2; it2 != name_soc.end(); it2++)
                {

                    my_content = "【" + it2->first + "】";
                    my_content = msg_pad(my_content, content_size);
                    SendMsg = from_nickname + to_nickname + time_stamp + my_content;
                    memset(SendBuf, '\0', sizeof(SendBuf));
                    strcpy(SendBuf, SendMsg.c_str());
                    send(Cli[i].sock, SendBuf, sizeof(SendBuf), NULL);


                }

                continue;
            }
            //消息转发处理
            map<string, SOCKET>::iterator it = name_soc.find(msg_rec.to_nickname);
            if (it == name_soc.end())
            {
                cout << "没有找到用户【" << msg_rec.to_nickname << "】" << "     <" << msg_rec.timestamp << ">" << endl;
                from_nickname = "serve";
                from_nickname = msg_pad(from_nickname, nickname_size);
                to_nickname = msg_rec.from_nickname;
                to_nickname = msg_pad(to_nickname, nickname_size);
                time(&timep);
                time_stamp = ctime(&timep);
                time_stamp = msg_pad(time_stamp, time_size);
                my_content = "抱歉，没有找到您想要发送的用户，用户不存在或者已下线。";
                my_content = msg_pad(my_content, content_size);
                SendMsg = from_nickname + to_nickname + time_stamp + my_content;
                memset(SendBuf, '\0', sizeof(SendBuf));
                strcpy(SendBuf, SendMsg.c_str());
                send(Cli[i].sock, SendBuf, sizeof(SendBuf), NULL);
                continue;

            }
            else
            {
                if (it->second != INVALID_SOCKET)
                {
                    send(it->second, RecBuf, sizeof(RecBuf), 0);
                    cout << "用户【" << msg_rec.from_nickname << "】发送给用户【" << msg_rec.to_nickname << "】的消息为：" << msg_rec.send_content << "     <" << msg_rec.timestamp << ">" << endl;
                }
                else
                {
                    name_soc.erase(it->first);
                    cout << "没有找到用户【" << msg_rec.to_nickname << "】" << "     <" << msg_rec.timestamp << ">" << endl;
                    from_nickname = "serve";
                    from_nickname = msg_pad(from_nickname, nickname_size);
                    to_nickname = msg_rec.from_nickname;
                    to_nickname = msg_pad(to_nickname, nickname_size);
                    time(&timep);
                    time_stamp = ctime(&timep);
                    time_stamp = msg_pad(time_stamp, time_size);
                    my_content = "抱歉，没有找到您想要发送的用户，用户不存在或者已下线。";
                    my_content = msg_pad(my_content, content_size);
                    SendMsg = from_nickname + to_nickname + time_stamp + my_content;
                    memset(SendBuf, '\0', sizeof(SendBuf));
                    strcpy(SendBuf, SendMsg.c_str());
                    send(Cli[i].sock, SendBuf, sizeof(SendBuf), NULL);
                }
                continue;
            }

        }

    }
    return 0;
}

DWORD WINAPI Server_listen_Thread(LPVOID lp)
{
    //接收到连接请求时首先生成一个新的套接口
    SOCKET soc = (SOCKET)lp;
    //进入监听
    listen(soc, 20);
    cout << "进入监听" << endl;
    while (1)
    {
        if(NOW_CLIENT_NUM < MAX_CLIENT_NUM)
            Cli[NOW_CLIENT_NUM].sock = accept(soc, (SOCKADDR*)&Cli[NOW_CLIENT_NUM].CliAddr, &nSize);
        if (Cli[NOW_CLIENT_NUM].sock != INVALID_SOCKET)
        {
            NOW_CLIENT_NUM++;
            //每个线程管理一个客户端套接口
            hListen_Thread[NOW_CLIENT_NUM - 1] = CreateThread(NULL, 0, Server_Rec_Thread, (LPVOID)(NOW_CLIENT_NUM - 1), 0, NULL);
        }
    }
    return 0;
}

int main() {
    //加载socket库
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0)
    {
        cout << "加载socket库出错";
        system("pause");
        return 0;
    }

    //创建套接字，第一个参数表示所使用的地址协议
    //第二个参数表示SOCKET类型，SOCK_STREAM用TCP协议则使用SOCK_STREAM，如果使用UDP，则对应参数为SOCK_DGRAM
    //第三个参数指定协议
    SOCKET servSock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
    //生成绑定socket所用的sockaddr_in结构体，并初始化IP地址和端口号
    sockaddr_in sockAddr;
    memset(&sockAddr, 0, sizeof(sockAddr));  //每个字节都用0填充
    sockAddr.sin_family = PF_INET;
    //sockAddr.sin_addr.s_addr = inet_addr("127.0.0.1");  //具体的IP地址
    inet_pton(PF_INET, "127.0.0.1", &sockAddr.sin_addr.s_addr);
    sockAddr.sin_port = htons(1234);  //端口号
    //绑定套接字，SOCKADDR将ip和端口放到同一个变量中，不易初始化，因此初始化用sockaddr_in结构
    bind(servSock, (SOCKADDR*)&sockAddr, sizeof(SOCKADDR));
    //LPVOID* LP_link = (LPVOID*)&servSock;
    HANDLE link_request = CreateThread(NULL, 0, Server_listen_Thread, (LPVOID)servSock, 0, NULL);


    while (1)
    {
        memset(InputBuf, '/0', BUF_SIZE);
        cin.getline(InputBuf, BUF_SIZE);
        if (strcmp(InputBuf, "/exit") == 0)
        {
            break;
        }

    }
    CloseHandle(link_request);
    //关闭套接字
    for (int i = 0; i < NOW_CLIENT_NUM; i++)
        closesocket(Cli[i].sock);
    closesocket(servSock);
    //终止 DLL 的使用
    WSACleanup();
    return 0;
}