#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#include<WinSock2.h>
#include<Ws2tcpip.h>
#include<cstring>
#include<string>
#include <map>
#include<time.h>
#pragma comment(lib, "ws2_32.lib")   //���� ws2_32.dll������Ŀ
using namespace std;


//���建����
const int BUF_SIZE = 1300;
const int nickname_size = 100;
const int content_size = 1000;
const int time_size = 100;
const int MAX_CLIENT_NUM = 20;
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
//�����ӵĿͻ��˵���Ϣ�ṹ��
struct CLIENT
{
    SOCKET sock;
    SOCKADDR_IN CliAddr;
    char nickname[100]="";
}Cli[MAX_CLIENT_NUM];

//��Ϣ�ṹ��
struct msg
{
    string from_nickname="";
    string to_nickname="";
    string timestamp = "";
    string send_content="";
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
    transpond_msg.timestamp = temp.substr(0, x-1);

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
                    cout << "�û���¼ʧ�ܣ����û�����"+ msg_rec.from_nickname +"���ѱ�ռ�á�" <<"     <"<<msg_rec.timestamp<<">"<< endl;
                    from_nickname = "serve";
                    from_nickname = msg_pad(from_nickname, nickname_size);
                    to_nickname = msg_rec.from_nickname;
                    to_nickname = msg_pad(to_nickname, nickname_size);
                    time(&timep);
                    time_stamp = ctime(&timep);
                    time_stamp = msg_pad(time_stamp, time_size);
                    my_content = "rename";
                    my_content = msg_pad(my_content, content_size);
                    SendMsg = from_nickname + to_nickname+ time_stamp+ my_content;
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
                cout << "��" << msg_rec.from_nickname << "��" << "�����ˣ�" <<"     <" << msg_rec.timestamp<<">"<<endl;
                from_nickname = "serve";
                from_nickname = msg_pad(from_nickname, nickname_size);
                to_nickname = string(Cli[i].nickname);
                to_nickname = msg_pad(to_nickname, nickname_size);
                time(&timep);
                time_stamp = ctime(&timep);
                time_stamp = msg_pad(time_stamp, time_size);
                my_content = "�������ӳɹ������ѳɹ���¼������ϵͳ��";
                my_content = msg_pad(my_content, content_size);
                SendMsg = from_nickname + to_nickname+time_stamp + my_content;
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
                    if (it1->first != msg_rec.from_nickname &&it1->second!=INVALID_SOCKET)
                    {
                        to_nickname = it1->first;
                        to_nickname = msg_pad(to_nickname, nickname_size);
                        SendMsg = from_nickname + to_nickname+time_stamp + my_content;
                        memset(SendBuf, '\0', sizeof(SendBuf));
                        strcpy(SendBuf, SendMsg.c_str());
                        send(it1->second, SendBuf, sizeof(SendBuf), NULL);
                    }

                }


                continue;
            }
            //����������
            if (strcmp(msg_rec.to_nickname.c_str(), "serve") == 0 && strcmp(msg_rec.send_content.c_str(), "quit") == 0)
            {
                closesocket(Cli[i].sock);
                cout << "��" << msg_rec.from_nickname << "��" << "������" <<"     <"<<msg_rec.timestamp<<">"<< endl;
                from_nickname = "serve";
                from_nickname = msg_pad(from_nickname, nickname_size);
                time(&timep);
                time_stamp = ctime(&timep);
                time_stamp = msg_pad(time_stamp, time_size);
                my_content = "�û���" + msg_rec.from_nickname + "�������ˡ�";
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
                        SendMsg = from_nickname + to_nickname+time_stamp + my_content;
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
                if(flag)
                    name_soc.erase(del_client);

                break;
            }
            //��ѯ�û�����
            if (strcmp(msg_rec.to_nickname.c_str(), "serve") == 0 && strcmp(msg_rec.send_content.c_str(), "online_list") == 0)
            {
                cout << "��" << msg_rec.from_nickname << "��" << "�����ѯ��������" << "     <"<<msg_rec.timestamp<<">"<<endl;

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
                        SendMsg = from_nickname + to_nickname+time_stamp + my_content;
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
                cout << "û���ҵ��û���" << msg_rec.to_nickname <<"��" << "     <" << msg_rec.timestamp << ">" << endl;
                from_nickname = "serve";
                from_nickname = msg_pad(from_nickname, nickname_size);
                to_nickname = msg_rec.from_nickname;
                to_nickname = msg_pad(to_nickname, nickname_size);
                time(&timep);
                time_stamp = ctime(&timep);
                time_stamp = msg_pad(time_stamp, time_size);
                my_content = "��Ǹ��û���ҵ�����Ҫ���͵��û����û������ڻ��������ߡ�";
                my_content = msg_pad(my_content, content_size);
                SendMsg = from_nickname + to_nickname+time_stamp + my_content;
                memset(SendBuf, '\0', sizeof(SendBuf));
                strcpy(SendBuf, SendMsg.c_str());
                send(Cli[i].sock, SendBuf, sizeof(SendBuf), NULL);

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
    cout << "�������"<<endl;
    while(1)
    {
        if(NOW_CLIENT_NUM < MAX_CLIENT_NUM)
            Cli[NOW_CLIENT_NUM].sock=accept(soc, (SOCKADDR*)&Cli[NOW_CLIENT_NUM].CliAddr, &nSize);
        if ( Cli[NOW_CLIENT_NUM].sock != INVALID_SOCKET)
        {
            NOW_CLIENT_NUM++;
            //ʹ��һ���߳̽��к�������������ɶ���ͬʱ���Ӵ�����ӵ��
            hListen_Thread[NOW_CLIENT_NUM-1] = CreateThread(NULL, 0, Server_Rec_Thread, (LPVOID)(NOW_CLIENT_NUM - 1), 0, NULL);
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
    // �ڶ���������ʾSOCKET���ͣ�SOCK_STREAM��TCPЭ����ʹ��SOCK_STREAM�����ʹ��UDP�����Ӧ����ΪSOCK_DGRAM
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
    for(int i=0;i<NOW_CLIENT_NUM;i++)
        closesocket(Cli[i].sock);
    closesocket(servSock);
    //��ֹ DLL ��ʹ��
    WSACleanup();
    return 0;
}