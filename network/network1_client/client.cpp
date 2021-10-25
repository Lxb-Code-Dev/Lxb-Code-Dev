#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#include<WinSock2.h>
#include <Ws2tcpip.h>
#include<cstring>
#include<string>
#include <time.h>
#pragma comment(lib, "ws2_32.lib")   //���� ws2_32.dll������Ŀ
using namespace std;

//���建����
const int BUF_SIZE = 1300;
const int nickname_size = 100;
const int content_size = 1000;
const int time_size = 100;
int offline = 0;//�Ƿ�����
int reNAME = 1;
char SendBuf[BUF_SIZE];
char RecBuf[BUF_SIZE];
char InputBuf[BUF_SIZE];
string from_nickname;
string to_nickname;
string time_stamp;
string my_content;
string SendMsg;
time_t timep;

//��Ϣ�ṹ��
struct msg
{
    string from_nickname = "";
    string to_nickname = "";
    string timestamp = "";
    string send_content = "";
};
//�����ַ�����亯�����Է�����Ϣ���зֶΣ�ÿ�����й̶���ʽ����#��β��ʣ�೤�����Ϊ�ո�
string msg_pad(string& msg, int length)
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

//����������Ϣ�̺߳���
DWORD WINAPI cli_recv(LPVOID param) {

    SOCKET c_socket = (SOCKET)(LPVOID)param;
    while (1) {
        memset(RecBuf, '\0', sizeof(RecBuf));
        int len_recv=recv(c_socket, RecBuf, sizeof(RecBuf), 0);
        if (len_recv > 0)
        {
            msg rec_msg = split_msg(RecBuf);
            if (strcmp(rec_msg.from_nickname.c_str(), "serve") == 0 && strcmp(rec_msg.send_content.c_str(), "rename") == 0)
            {
                cout << "��������ʧ�ܣ����û����ѱ�ռ�ã�" << "     <" << rec_msg.timestamp << ">" << endl;
                cout << "���������������û�����";
                getline(cin, from_nickname);

                from_nickname = msg_pad(from_nickname, nickname_size);
                to_nickname = "serve";
                to_nickname = msg_pad(to_nickname, nickname_size);
                time(&timep);
                time_stamp = ctime(&timep);
                time_stamp = msg_pad(time_stamp, time_size);
                my_content = "link_serve";
                my_content = msg_pad(my_content, content_size);
                SendMsg = from_nickname + to_nickname +time_stamp+ my_content;
                memset(SendBuf, '\0', sizeof(SendBuf));
                strcpy(SendBuf, SendMsg.c_str());
                send(c_socket, SendBuf, sizeof(SendBuf), NULL);
                continue;
            }
            if (strcmp(rec_msg.from_nickname.c_str(), "serve") == 0 && strcmp(rec_msg.send_content.c_str(), "confirm") == 0)
            {
                cout << "�û������ã���½ע����ɡ�" << "     <" << rec_msg.timestamp << ">" << endl;
                reNAME = 0;
                continue;
            }
            //reNAME = 0;
            cout << "��" << rec_msg.from_nickname << "����";
            cout << rec_msg.send_content <<"     <"<<rec_msg.timestamp<<">"<< endl;
            
        }
    }
    return 0;
}

int main() 
{
    //����socket��
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0)
    {
        cout << "����socket�����";
        system("pause");
        return 0;
    }
    //�����׽���
    SOCKET Clisock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
    //��������ַ��Ϣ
    sockaddr_in sockAddr;
    memset(&sockAddr, 0, sizeof(sockAddr));  //ÿ���ֽڶ���0���
    sockAddr.sin_family = PF_INET;
    //sockAddr.sin_addr.s_addr = inet_addr("127.0.0.1");
    inet_pton(PF_INET, "127.0.0.1", &sockAddr.sin_addr.s_addr);
    sockAddr.sin_port = htons(1234);
    cout << "���ڳ������ӷ�����"<<endl;
    while (1)
    {
        if (connect(Clisock, (SOCKADDR*)&sockAddr, sizeof(SOCKADDR)) != SOCKET_ERROR)
        {
            cout << "���ӷ������ɹ�����ӭ���������ҡ�" << endl;
            cout << "����������ǳƣ�";
            getline(cin, from_nickname);
            
            from_nickname = msg_pad(from_nickname, nickname_size);
          
            to_nickname = "serve";
            to_nickname = msg_pad(to_nickname, nickname_size);
            time(&timep); 
            time_stamp=ctime(&timep);
            time_stamp = msg_pad(time_stamp, time_size);
            my_content = "link_serve";
            my_content = msg_pad(my_content, content_size);
            SendMsg = from_nickname + to_nickname +time_stamp+ my_content;
            memset(SendBuf, '\0', sizeof(SendBuf));
            strcpy(SendBuf, SendMsg.c_str());
            send(Clisock, SendBuf, sizeof(SendBuf), NULL);
            
            break;
        }
    }

    
    LPVOID LP_recv = (LPVOID)Clisock;
    HANDLE recv_request = CreateThread(NULL, 0, cli_recv, LP_recv, 0, NULL);
    
    while (1)
    {
        while (1)
        {
            if (!reNAME)
                break;

        }
        getline(cin, to_nickname);
        getline(cin, my_content);
        if (strcmp(to_nickname.c_str(), "serve") == 0 && strcmp(my_content.c_str(), "quit") == 0)
        {
            offline = 1;
        }
        while (to_nickname.length() >= nickname_size)
        {
            cout << "���뷢���˴�������������:";
            to_nickname = "";
            getline(cin, to_nickname);
        }
        while (my_content.length() >= content_size)
        {
            cout << "������Ϣ�������ƣ�����������:";
            my_content = "";
            getline(cin, my_content);
        }
        time(&timep);
        time_stamp = ctime(&timep);
        time_stamp = msg_pad(time_stamp, time_size);
        to_nickname = msg_pad(to_nickname, nickname_size);
        my_content = msg_pad(my_content, content_size);
        SendMsg = from_nickname + to_nickname + time_stamp + my_content;
        memset(SendBuf, '\0', sizeof(SendBuf));
        strcpy(SendBuf,SendMsg.c_str() );
        send(Clisock, SendBuf, BUF_SIZE, NULL);
        if (offline)
        {
            return 0;
        }
        
    }
    CloseHandle(recv_request);
    //�ر��׽���
    closesocket(Clisock);
    //��ֹʹ�� DLL
    WSACleanup();
    system("pause");
    return 0;
}