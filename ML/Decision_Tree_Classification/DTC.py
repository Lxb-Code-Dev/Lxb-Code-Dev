import pandas as pd
import math
import copy
train_data1=pd.read_csv("Watermelon-train1.csv",encoding="gbk",header=None).values.tolist()
test_data1=pd.read_csv("Watermelon-test1.csv",encoding="gbk",header=None).values.tolist()
train_data2=pd.read_csv("Watermelon-train2.csv",encoding="gbk",header=None).values.tolist()
test_data2=pd.read_csv("Watermelon-test2.csv",encoding="gbk",header=None).values.tolist()
train_data1=[i[1:] for i in train_data1]
train_data2=[i[1:] for i in train_data2]
test_data1=[i[1:] for i in test_data1]
test_data2=[i[1:] for i in test_data2]

class DTC():
    def __init__(self,train,test):
        '''
        :param train: 训练集
        :param test: 测试集
        '''
        self.train=train
        self.test=test
        self.tree={}
        self.score=0
        self.trueclass=[]
        self.predictclass=[]
    def ID3(self,father,DATA):
        '''
        递归构造ID3树，以字典形式放入self.tree中
        :return:
        '''
        data = DATA[1:]
        #没有剩余属性或者属性全相同
        if len(DATA[0])==1 or len([i for i in data if i!=data[0]])==0:
            kd=[i[-1] for i in data]
            max_kd=max(kd,key=kd.count)
            father[DATA[0][-1]]=max_kd  #产生叶子节点
            return
        D = [i[-1] for i in data]
        kinds = set(D)
        #只剩一个分类
        if len(kinds)==1:
            father[DATA[0][-1]]=D[0]
            return
        num = len(data)
        info=[]
        num1=len(data[0])-1
        for i in range(num1):
            tupleD = [(k[i],k[-1]) for k in data]
            D=[k[i] for k in data]
            kinds = set(D)
            info_one = 0
            for kind in kinds:
                p1 = D.count(kind) / num
                A=[k[1] for k in tupleD if k[0]==kind]
                A_num=len(A)
                A_kind=set(A)
                info_temp=0
                for j in A_kind:
                    p2=A.count(j)/A_num
                    info_temp-=p2*math.log(p2,2)
                info_one+=p1*info_temp
            info.append(info_one)
        #选择下一个节点代表的属性
        node_index=info.index(min(info))
        tupleD=[k[node_index] for k in data]
        kinds=set(tupleD)
        for kind in kinds:
            new_sample=[DATA[0]]
            temp=[k for k in data if k[node_index]==kind]
            new_sample.extend(temp)
            res=[]
            for sample in new_sample:
                a=sample[0:node_index]
                b=sample[node_index+1:]
                a.extend(b)
                res.append(a)
            father[kind]={}
            self.ID3(father[kind],res)
    def C4_5(self,father,DATA):
        '''
        递归构造C4.5树，以字典形式放入self.tree中
        :return:
        '''
        data = DATA[1:]
        # 没有剩余属性或者属性全相同
        if len(DATA[0]) == 1 or len([i for i in data if i != data[0]]) == 0:
            #print(DATA)
            kd = [i[-1] for i in data]
            #print(kd)
            max_kd = max(kd, key=kd.count)
            father[DATA[0][-1]] = max_kd  # 产生叶子节点
            return
        D = [i[-1] for i in data]
        kinds = set(D)
        # 只剩一个分类
        if len(kinds) == 1:
            father[DATA[0][-1]] = D[0]
            return
        num = len(data)
        info = []
        num1 = len(data[0]) - 1
        max_index=None
        for i in range(num1):
            if DATA[0][i]=='密度':
                #如果是连续值，就从n-1个二分点中取一个最好的
                gr = []
                for j in range(num-1):
                    # print(data[j][i])
                    # print(data[j+1][i])
                    mid = (eval(data[j][i]) + eval(data[j+1][i])) / 2
                    p1 = (j + 1) / num
                    tt=0
                    for m in range(2):
                        temp = []
                        if m==1:
                            p1=1-p1
                            temp=data[j+1:]
                        else:
                            temp=data[:j+1]
                        kind_gr=[x[-1] for x in temp]
                        kind_gr_set=set(kind_gr)
                        info_temp = 0
                        for kk in kind_gr_set:
                            p2=kind_gr.count(kk)/len(kind_gr)
                            info_temp -= p2 * math.log(p2, 2)
                        tt+=p1*info_temp
                    gr.append(tt)
                max_index=gr.index(max(gr))
                info.append(max(gr))
                continue

            tupleD = [(k[i], k[-1]) for k in data]
            D = [k[i] for k in data]
            kinds = set(D)
            info_one = 0
            iv=0
            for kind in kinds:
                p1 = D.count(kind) / num
                iv-=p1 * math.log(p1, 2)
                A = [k[1] for k in tupleD if k[0] == kind]
                A_num = len(A)
                A_kind = set(A)
                info_temp = 0
                for j in A_kind:
                    p2 = A.count(j) / A_num
                    info_temp -= p2 * math.log(p2, 2)
                info_one += p1 * info_temp
            info.append(info_one/iv)

        # 选择下一个节点代表的属性
        node_index = info.index(max(info))
        if DATA[0][node_index]=='密度':
            #二分
            q=(eval(data[max_index][node_index])+eval(data[max_index+1][node_index]))/2
            new_sample = [DATA[0]]
            temp=data[:max_index+1]
            new_sample.extend(temp)
            res = []
            for sample in new_sample:
                a = sample[0:node_index]
                b = sample[node_index + 1:]
                a.extend(b)
                res.append(a)
            father['小于等于_' + str(q)] = {}
            self.C4_5(father['小于等于_'+str(q)], res)
            new_sample1 = [DATA[0]]
            temp1 = data[:max_index + 1]
            new_sample1.extend(temp1)
            res1 = []
            for sample1 in new_sample1:
                a = sample1[0:node_index]
                b = sample1[node_index + 1:]
                a.extend(b)
                res1.append(a)
            father['大于_' + str(q)] = {}
            self.C4_5(father['大于_' + str(q)], res1)
        else:
            tupleD = [k[node_index] for k in data]
            kinds = set(tupleD)
            for kind in kinds:
                new_sample = [DATA[0]]
                temp = [k for k in data if k[node_index] == kind]
                new_sample.extend(temp)
                res = []
                for sample in new_sample:
                    a = sample[0:node_index]
                    b = sample[node_index + 1:]
                    a.extend(b)
                    res.append(a)
                father[kind] = {}
                self.C4_5(father[kind], res)

    def is_number(self,s):
        '''
        :param s: 要判断的字符串
        :return: 返回是否为数字
        '''
        try:
            float(s)
            return True
        except ValueError:
            pass
        try:
            import unicodedata  # 处理ASCii码的包
            unicodedata.numeric(s)  # 把一个表示数字的字符串转换为浮点数返回的函数
            return True
        except (TypeError, ValueError):
            pass
        return False

    def predict(self,testdata,node_dict):
        '''
        根据构建的树self.tree对测试集进行预测
        :return: 返回测试结果，[]
        '''
        flag=0
        first=list(node_dict.keys())
        if '好瓜' in first:
            #print(node_dict['好瓜'])
            if testdata[-1][-1]==node_dict['好瓜']:
                self.score+=1
            self.trueclass.append(testdata[-1][-1])
            self.predictclass.append(node_dict['好瓜'])
            #print("实际值：",testdata[-1][-1])
            #print("预测值：",node_dict['好瓜'])
            return
        if len(first[0].split('_'))==2:
            flag=1
        for samp in testdata:
            if flag:
                a = samp.copy()
                k = [i for i in samp if self.is_number(i)]
                if eval(k[0])<=eval(first[0].split('_')[1]):
                    a.remove(k[0])
                    self.predict([a], node_dict['小于等于_'+first[0].split('_')[1]])
                else:
                    a.remove(k[0])
                    self.predict([a], node_dict['大于_' + first[0].split('_')[1]])
            else:
                a=samp.copy()
                k=[i for i in samp if i in first]
                a.remove(k[0])
                self.predict([a],node_dict[k[0]])
    def run1(self):
        self.ID3(self.tree,self.train)
        testdata = self.test[1:]
        self.predict(testdata,self.tree)
        print("实际结果为：",self.trueclass)
        print("预测结果为：", self.predictclass)
        print("ID3预测精度为：",self.score/len(testdata))
    def run2(self):
        self.C4_5(self.tree,self.train)
        #print(self.tree)
        testdata = self.test[1:]
        #为了方便连续值的处理，将数据按照连续值这一列进行升序排列
        testdata.sort(key=lambda x: x[-2], reverse=False)
        self.predict(testdata, self.tree)
        print("实际结果为：", self.trueclass)
        print("预测结果为：", self.predictclass)
        print("C4.5算法预测精度为：", self.score / len(testdata))

if __name__=="__main__":
    dtc1=DTC(train_data1,test_data1)
    dtc1.run1()
    dtc2=DTC(train_data2,test_data2)
    dtc2.run2()

