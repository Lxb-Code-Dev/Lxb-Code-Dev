#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#include<WinSock2.h>
#include<Ws2tcpip.h>
#include<cstring>
#include<string>
#include <map>
#include<time.h>
#include<set>//ʵ��Ⱥ��
#pragma comment(lib, "ws2_32.lib")   //���� ws2_32.dll������Ŀ
using namespace std;


//���建����
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
//ÿ���ͻ��˰���һ���̴߳���
HANDLE hListen_Thread[MAX_CLIENT_NUM];
HANDLE hRecv_Thread[MAX_CLIENT_NUM];
map<string, SOCKET> name_soc;
//Ⱥ�Ľṹ��<Ⱥ������Ա����>
map<int, set<SOCKET>> group_set_map;
//�����ӵĿͻ��˵���Ϣ�ṹ��
struct CLIENT
{
    SOCKET sock;
    SOCKADDR_IN CliAddr;
    char nickname[100] = "";
}Cli[MAX_CLIENT_NUM];

//��Ϣ�ṹ��
struct msg
{
    string from_nickname = "";
    string to_nickname = "";
    string timestamp = "";
    string send_content = "";
};




//�����ַ�����亯�����Է�����Ϣ���зֶΣ�ÿ�����й̶���ʽ����#��β��ʣ�೤�����Ϊ�ո�
string msg_pad(string msg, int length)
{
    int round = length - msg.length();
    msg += '#';
    for (int i = 1; i < round; i++) {
        msg += " ";
    }
    return msg;
}
//���յ�����Ϣ���մ���Э����в��
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
//������Ϣ�����߳�
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
            //���ε�¼�Ǽ�����
            if (strcmp(msg_rec.to_nickname.c_str(), "serve") == 0 && strcmp(msg_rec.send_content.c_str(), "link_serve") == 0)
            {

                map<string, SOCKET>::iterator it = name_soc.find(msg_rec.from_nickname);
                if (it != name_soc.end() || strcmp(msg_rec.from_nickname.c_str(), "serve") == 0)
                {
                    cout << "�û���¼ʧ�ܣ����û�����" + msg_rec.from_nickname + "���ѱ�ռ�á�" << "     <" << msg_rec.timestamp << ">" << endl;
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
                //ȷ��ע��
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
                cout << "��" << msg_rec.from_nickname << "��" << "�����ˣ�" << "     <" << msg_rec.timestamp << ">" << endl;
                from_nickname = "serve";
                from_nickname = msg_pad(from_nickname, nickname_size);
                to_nickname = string(Cli[i].nickname);
                to_nickname = msg_pad(to_nickname, nickname_size);
                time(&timep);
                time_stamp = ctime(&timep);
                time_stamp = msg_pad(time_stamp, time_size);
                my_content = "�������ӳɹ������ѳɹ���¼������ϵͳ��";
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
                my_content = "�û���" + msg_rec.from_nickname + "�������ˡ�";
                my_content = msg_pad(my_content, content_size);
                //֪ͨ�����û�������������
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
            //���봴��Ⱥ��
            if (strcmp(msg_rec.to_nickname.c_str(), "serve") == 0 && strcmp(msg_rec.send_content.c_str(), "create_group") == 0)
            {
                set<SOCKET> sok_set;
                sok_set.insert(Cli[i].sock);
                group_set_map.insert(pair<int, set<SOCKET>>(group_num, sok_set));
                cout<<"��"<<msg_rec.from_nickname<<"������Ⱥ�ģ�����������Ⱥ��Ϊ"<< group_num<< "     <" << msg_rec.timestamp << ">" << endl;
                from_nickname = "serve";
                from_nickname = msg_pad(from_nickname, nickname_size);
                to_nickname = string(Cli[i].nickname);
                to_nickname = msg_pad(to_nickname, nickname_size);
                time(&timep);
                time_stamp = ctime(&timep);
                time_stamp = msg_pad(time_stamp, time_size);
                my_content = "Ⱥ�Ĵ����ɹ���Ⱥ��Ϊ"+to_string(group_num);
                my_content = msg_pad(my_content, content_size);
                SendMsg = from_nickname + to_nickname + time_stamp + my_content;
                memset(SendBuf, '\0', sizeof(SendBuf));
                strcpy(SendBuf, SendMsg.c_str());
                send(Cli[i].sock, SendBuf, sizeof(SendBuf), NULL);
                group_num++;
                continue;
            }
            //��Ⱥ�﷢��Ϣ������Ⱥ��ģʽ
            if (strcmp(msg_rec.to_nickname.c_str(), "serve") == 0 && strcmp(msg_rec.send_content.c_str(), "group_chat") == 0)
            {
                cout<<"��"<< msg_rec.from_nickname<<"������Ⱥ��ģʽ=========="<< "     <" << msg_rec.timestamp << ">" << endl;
                from_nickname = "serve";
                from_nickname = msg_pad(from_nickname, nickname_size);
                to_nickname = string(Cli[i].nickname);
                to_nickname = msg_pad(to_nickname, nickname_size);
                time(&timep);
                time_stamp = ctime(&timep);
                time_stamp = msg_pad(time_stamp, time_size);
                my_content = "���ѽ���Ⱥ��ģʽ==========" ;
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
                        //Ⱥ����Ϣ
                        msg grmsg_rec = split_msg(RecBuf);
                        //�˳�����
                        if (strcmp(grmsg_rec.to_nickname.c_str(), "serve") == 0 && strcmp(grmsg_rec.send_content.c_str(), "quit") == 0)
                        {
                            closesocket(Cli[i].sock);
                            cout << "��" << grmsg_rec.from_nickname << "��" << "������" << "     <" << grmsg_rec.timestamp << ">" << endl;
                            from_nickname = "serve";
                            from_nickname = msg_pad(from_nickname, nickname_size);
                            time(&timep);
                            time_stamp = ctime(&timep);
                            time_stamp = msg_pad(time_stamp, time_size);
                            my_content = "�û���" + grmsg_rec.from_nickname + "�������ˡ�";
                            my_content = msg_pad(my_content, content_size);
                            //֪ͨ�����û�������������
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
                        //�˳�Ⱥ��
                        if (strcmp(grmsg_rec.to_nickname.c_str(), "serve") == 0 && strcmp(grmsg_rec.send_content.c_str(), "quit_group") == 0)
                        {
                            cout << "��" << grmsg_rec.from_nickname << "���˳�Ⱥ��ģʽ==========" << "     <" << grmsg_rec.timestamp << ">" << endl;
                            from_nickname = "serve";
                            from_nickname = msg_pad(from_nickname, nickname_size);
                            to_nickname = string(Cli[i].nickname);
                            to_nickname = msg_pad(to_nickname, nickname_size);
                            time(&timep);
                            time_stamp = ctime(&timep);
                            time_stamp = msg_pad(time_stamp, time_size);
                            my_content = "�����˳�Ⱥ��ģʽ==========";
                            my_content = msg_pad(my_content, content_size);
                            SendMsg = from_nickname + to_nickname + time_stamp + my_content;
                            memset(SendBuf, '\0', sizeof(SendBuf));
                            strcpy(SendBuf, SendMsg.c_str());
                            send(Cli[i].sock, SendBuf, sizeof(SendBuf), NULL);
                            break;
                        }
                        if (to_string(atoi(grmsg_rec.to_nickname.c_str())) != grmsg_rec.to_nickname)
                        {
                            cout << "û���ҵ�Ⱥ�ġ�" << grmsg_rec.to_nickname << "��" << "     <" << msg_rec.timestamp << ">" << endl;
                            from_nickname = "serve";
                            from_nickname = msg_pad(from_nickname, nickname_size);
                            to_nickname = grmsg_rec.from_nickname;
                            to_nickname = msg_pad(to_nickname, nickname_size);
                            time(&timep);
                            time_stamp = ctime(&timep);
                            time_stamp = msg_pad(time_stamp, time_size);
                            my_content = "��Ǹ��û�ж�ӦȺ�ģ����������Ƿ���ȷ��";
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
                            cout << "û���ҵ�Ⱥ�ġ�" << grmsg_rec.to_nickname << "��" << "     <" << msg_rec.timestamp << ">" << endl;
                            from_nickname = "serve";
                            from_nickname = msg_pad(from_nickname, nickname_size);
                            to_nickname = grmsg_rec.from_nickname;
                            to_nickname = msg_pad(to_nickname, nickname_size);
                            time(&timep);
                            time_stamp = ctime(&timep);
                            time_stamp = msg_pad(time_stamp, time_size);
                            my_content = "��Ǹ��û�ж�ӦȺ�ģ����������Ƿ���ȷ��";
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
                                my_content = "��Ǹ����δ�����Ⱥ�ģ�����ʧ�ܡ�";
                                my_content = msg_pad(my_content, content_size);
                                SendMsg = from_nickname + to_nickname + time_stamp + my_content;
                                memset(SendBuf, '\0', sizeof(SendBuf));
                                strcpy(SendBuf, SendMsg.c_str());
                                send(Cli[i].sock, SendBuf, sizeof(SendBuf), NULL);
                                cout << "�û���" << grmsg_rec.from_nickname << "��δ����Ⱥ�ġ�" << grmsg_rec.to_nickname << "����������Ϣʧ�ܡ�" << "     <" << msg_rec.timestamp << ">" << endl;
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
                                cout << "��" << grmsg_rec.from_nickname << "����Ⱥ�ġ�"<< grmsg_rec.to_nickname<<"���з��ͣ�" <<grmsg_rec.send_content<< "     <" << msg_rec.timestamp << ">" << endl;
                                continue;
                            }
                        }
                        
                    }
                }
                if (offline)
                    break;
                continue;
                
            }
            //����Ⱥ��
            if (strcmp(msg_rec.to_nickname.c_str(), "serve") == 0 && strcmp(msg_rec.send_content.substr(0,10).c_str(), "join_group") == 0)
            {
                int group_n = atoi(msg_rec.send_content.substr(10, msg_rec.send_content.length()).c_str());
                if (to_string(group_n) != msg_rec.send_content.substr(10, msg_rec.send_content.length()))
                {
                    cout << "δ�ҵ�Ⱥ��Ϊ" << msg_rec.send_content.substr(10, msg_rec.send_content.length()) << "��Ⱥ�ġ�" << "     <" << msg_rec.timestamp << ">" << endl;
                    from_nickname = "serve";
                    from_nickname = msg_pad(from_nickname, nickname_size);
                    to_nickname = string(Cli[i].nickname);
                    to_nickname = msg_pad(to_nickname, nickname_size);
                    time(&timep);
                    time_stamp = ctime(&timep);
                    time_stamp = msg_pad(time_stamp, time_size);
                    my_content = "����Ⱥ��ʧ�ܣ�δ�ҵ�Ⱥ�ġ�";
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
                    cout<<"δ�ҵ�Ⱥ��Ϊ"<< msg_rec.send_content.substr(10, msg_rec.send_content.length())<<"��Ⱥ�ġ�"<< "     <" << msg_rec.timestamp << ">" << endl;
                    from_nickname = "serve";
                    from_nickname = msg_pad(from_nickname, nickname_size);
                    to_nickname = string(Cli[i].nickname);
                    to_nickname = msg_pad(to_nickname, nickname_size);
                    time(&timep);
                    time_stamp = ctime(&timep);
                    time_stamp = msg_pad(time_stamp, time_size);
                    my_content = "����Ⱥ��ʧ�ܣ�δ�ҵ�Ⱥ�ġ�" ;
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
                    cout << "��"<<msg_rec.from_nickname<<"������Ⱥ��" << msg_rec.send_content.substr(10, msg_rec.send_content.length()) << "��" << "     <" << msg_rec.timestamp << ">" << endl;
                    from_nickname = "serve";
                    from_nickname = msg_pad(from_nickname, nickname_size);
                    to_nickname = string(Cli[i].nickname);
                    to_nickname = msg_pad(to_nickname, nickname_size);
                    time(&timep);
                    time_stamp = ctime(&timep);
                    time_stamp = msg_pad(time_stamp, time_size);
                    my_content = "��"+ msg_rec.from_nickname +"������Ⱥ��"+ msg_rec.send_content.substr(10, msg_rec.send_content.length());
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
            //����������
            if (strcmp(msg_rec.to_nickname.c_str(), "serve") == 0 && strcmp(msg_rec.send_content.c_str(), "quit") == 0)
            {
                closesocket(Cli[i].sock);
                cout << "��" << msg_rec.from_nickname << "��" << "������" << "     <" << msg_rec.timestamp << ">" << endl;
                from_nickname = "serve";
                from_nickname = msg_pad(from_nickname, nickname_size);
                time(&timep);
                time_stamp = ctime(&timep);
                time_stamp = msg_pad(time_stamp, time_size);
                my_content = "�û���" + msg_rec.from_nickname + "�������ˡ�";
                my_content = msg_pad(my_content, content_size);
                map<int, set<SOCKET>>::iterator it6 = group_set_map.begin();
                for (it6; it6 != group_set_map.end(); it6++)
                {
                    if (it6->second.count(Cli[i].sock) != 0)
                        it6->second.erase(Cli[i].sock);
                }
                //֪ͨ�����û�������������
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
            //��ѯȺ�鴦��
            if (strcmp(msg_rec.to_nickname.c_str(), "serve") == 0 && strcmp(msg_rec.send_content.c_str(), "group_list") == 0)
            {
                cout << "��" << msg_rec.from_nickname << "��" << "�����ѯȺ������" << "     <" << msg_rec.timestamp << ">" << endl;

                from_nickname = "serve";
                from_nickname = msg_pad(from_nickname, nickname_size);
                to_nickname = msg_rec.from_nickname;
                to_nickname = msg_pad(to_nickname, nickname_size);
                time(&timep);
                time_stamp = ctime(&timep);
                time_stamp = msg_pad(time_stamp, time_size);
                //�ظ�Ⱥ���б�
                map<int, set<SOCKET>>::iterator it7 = group_set_map.begin();
                for (it7; it7 != group_set_map.end(); it7++)
                {

                    my_content = "��" + to_string(it7->first) + "��";
                    my_content = msg_pad(my_content, content_size);
                    SendMsg = from_nickname + to_nickname + time_stamp + my_content;
                    memset(SendBuf, '\0', sizeof(SendBuf));
                    strcpy(SendBuf, SendMsg.c_str());
                    send(Cli[i].sock, SendBuf, sizeof(SendBuf), NULL);


                }

                continue;
            }
            //��ѯ�û�����
            if (strcmp(msg_rec.to_nickname.c_str(), "serve") == 0 && strcmp(msg_rec.send_content.c_str(), "online_list") == 0)
            {
                cout << "��" << msg_rec.from_nickname << "��" << "�����ѯ��������" << "     <" << msg_rec.timestamp << ">" << endl;

                from_nickname = "serve";
                from_nickname = msg_pad(from_nickname, nickname_size);
                to_nickname = msg_rec.from_nickname;
                to_nickname = msg_pad(to_nickname, nickname_size);
                time(&timep);
                time_stamp = ctime(&timep);
                time_stamp = msg_pad(time_stamp, time_size);
                //�ظ������û��б�
                map<string, SOCKET>::iterator it2 = name_soc.begin();
                for (it2; it2 != name_soc.end(); it2++)
                {

                    my_content = "��" + it2->first + "��";
                    my_content = msg_pad(my_content, content_size);
                    SendMsg = from_nickname + to_nickname + time_stamp + my_content;
                    memset(SendBuf, '\0', sizeof(SendBuf));
                    strcpy(SendBuf, SendMsg.c_str());
                    send(Cli[i].sock, SendBuf, sizeof(SendBuf), NULL);


                }

                continue;
            }
            //��Ϣת������
            map<string, SOCKET>::iterator it = name_soc.find(msg_rec.to_nickname);
            if (it == name_soc.end())
            {
                cout << "û���ҵ��û���" << msg_rec.to_nickname << "��" << "     <" << msg_rec.timestamp << ">" << endl;
                from_nickname = "serve";
                from_nickname = msg_pad(from_nickname, nickname_size);
                to_nickname = msg_rec.from_nickname;
                to_nickname = msg_pad(to_nickname, nickname_size);
                time(&timep);
                time_stamp = ctime(&timep);
                time_stamp = msg_pad(time_stamp, time_size);
                my_content = "��Ǹ��û���ҵ�����Ҫ���͵��û����û������ڻ��������ߡ�";
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
                    cout << "�û���" << msg_rec.from_nickname << "�����͸��û���" << msg_rec.to_nickname << "������ϢΪ��" << msg_rec.send_content << "     <" << msg_rec.timestamp << ">" << endl;
                }
                else
                {
                    name_soc.erase(it->first);
                    cout << "û���ҵ��û���" << msg_rec.to_nickname << "��" << "     <" << msg_rec.timestamp << ">" << endl;
                    from_nickname = "serve";
                    from_nickname = msg_pad(from_nickname, nickname_size);
                    to_nickname = msg_rec.from_nickname;
                    to_nickname = msg_pad(to_nickname, nickname_size);
                    time(&timep);
                    time_stamp = ctime(&timep);
                    time_stamp = msg_pad(time_stamp, time_size);
                    my_content = "��Ǹ��û���ҵ�����Ҫ���͵��û����û������ڻ��������ߡ�";
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
    //���յ���������ʱ��������һ���µ��׽ӿ�
    SOCKET soc = (SOCKET)lp;
    //�������
    listen(soc, 20);
    cout << "�������" << endl;
    while (1)
    {
        if(NOW_CLIENT_NUM < MAX_CLIENT_NUM)
            Cli[NOW_CLIENT_NUM].sock = accept(soc, (SOCKADDR*)&Cli[NOW_CLIENT_NUM].CliAddr, &nSize);
        if (Cli[NOW_CLIENT_NUM].sock != INVALID_SOCKET)
        {
            NOW_CLIENT_NUM++;
            //ÿ���̹߳���һ���ͻ����׽ӿ�
            hListen_Thread[NOW_CLIENT_NUM - 1] = CreateThread(NULL, 0, Server_Rec_Thread, (LPVOID)(NOW_CLIENT_NUM - 1), 0, NULL);
        }
    }
    return 0;
}

int main() {
    //����socket��
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0)
    {
        cout << "����socket�����";
        system("pause");
        return 0;
    }

    //�����׽��֣���һ��������ʾ��ʹ�õĵ�ַЭ��
    //�ڶ���������ʾSOCKET���ͣ�SOCK_STREAM��TCPЭ����ʹ��SOCK_STREAM�����ʹ��UDP�����Ӧ����ΪSOCK_DGRAM
    //����������ָ��Э��
    SOCKET servSock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
    //���ɰ�socket���õ�sockaddr_in�ṹ�壬����ʼ��IP��ַ�Ͷ˿ں�
    sockaddr_in sockAddr;
    memset(&sockAddr, 0, sizeof(sockAddr));  //ÿ���ֽڶ���0���
    sockAddr.sin_family = PF_INET;
    //sockAddr.sin_addr.s_addr = inet_addr("127.0.0.1");  //�����IP��ַ
    inet_pton(PF_INET, "127.0.0.1", &sockAddr.sin_addr.s_addr);
    sockAddr.sin_port = htons(1234);  //�˿ں�
    //���׽��֣�SOCKADDR��ip�Ͷ˿ڷŵ�ͬһ�������У����׳�ʼ������˳�ʼ����sockaddr_in�ṹ
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
    //�ر��׽���
    for (int i = 0; i < NOW_CLIENT_NUM; i++)
        closesocket(Cli[i].sock);
    closesocket(servSock);
    //��ֹ DLL ��ʹ��
    WSACleanup();
    return 0;
}