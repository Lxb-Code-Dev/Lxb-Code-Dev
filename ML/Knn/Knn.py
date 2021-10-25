import pandas as pd
import math
import matplotlib.pyplot as plt
import copy
from sklearn.neighbors import KNeighborsClassifier
train_data=pd.read_csv("semeion_train.csv",header=None)
test_data=pd.read_csv("semeion_test.csv",header=None)

#下面两行代码让plt画图时可以显示中文
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False


class knn():
    def __init__(self,train,test,k,dises=[]):
        '''

        :param train: 训练集
        :param test: 测试集
        :param k: k值
        :param dises: 距离矩阵，默认为空
        '''
        self.train_data=train
        self.test_data=test
        self.k=k
        self.dises=dises

    def data_split(self,data):
        '''
        :param data: 数据集
        :return: 特征值，目标值
        '''
        data = data.values.tolist()
        for i in range(len(data)):
            data[i] = data[i][0].split(' ')
            data[i].pop(-1)
            data[i] = [eval(j) for j in data[i]]

        feature = [data[i][:-10] for i in range(0, len(data))]
        target = [data[i][-10:] for i in range(0, len(data))]
        return feature, target

    def distance(self,train,test):
        '''
        :param train: 留一法中剩余的向量
        :param test: 留一法中的一
        :return: 距离向量
        '''
        dis=[]
        for i in train:
            dis.append(math.sqrt(sum([(j-z)**2 for j,z in zip(i,test)])))
        return dis
    def result(self):
        '''
        :return: 准确率，距离矩阵
        '''
        train_feature, train_target = self.data_split(self.train_data)
        test_feature, test_target = self.data_split(self.test_data)

        correct = 0
        flag=0
        if self.dises==[]:
            for one in test_feature:
                self.dises.append(self.distance(train_feature,one))
        else:
            flag=1

        for z,i in enumerate(self.dises):
            k_neighbor = []
            temp=i.copy()


            for j in range(self.k):
                min_value=min(temp)
                k_neighbor.append(train_target[i.index(min_value)])
                temp.pop(temp.index(min_value))

            neighbor_kind=[i.index(1) for i in k_neighbor]
            max_neighbor=max(neighbor_kind,key=neighbor_kind.count)
            if max_neighbor==test_target[z].index(1):
                correct+=1

            else:
                pass
        if flag:
            return correct / len(self.test_data)
        else:
            return correct/len(self.test_data),self.dises

    def sklearn_knn(self):
        '''

        :return: knn分类器的准确率
        '''
        train_feature, train_target = self.data_split(self.train_data)
        test_feature, test_target = self.data_split(self.test_data)
        train_target=[i.index(1) for i in train_target]
        test_target=[i.index(1) for i in test_target]
        KNN=KNeighborsClassifier(n_neighbors=self.k)
        KNN.fit(train_feature,train_target)
        accuracy=KNN.score(test_feature,test_target)
        return accuracy
my_corr=['','','']
knn_corr=['','','']
KNN=knn(train_data,test_data,1)
my_corr[0],dises=KNN.result()
knn_corr[0]=KNN.sklearn_knn()
print("n=1时，手动knn正确率为：",my_corr[0])
print("n=1时，调包knn正确率为：",knn_corr[0])

KNN=knn(train_data,test_data,3,dises)
my_corr[1]=KNN.result()
knn_corr[1]=KNN.sklearn_knn()
print("n=3时，手动knn正确率为：",my_corr[1])
print("n=3时，调包knn正确率为：",knn_corr[1])

KNN=knn(train_data,test_data,5,dises)
my_corr[2]=KNN.result()
knn_corr[2]=KNN.sklearn_knn()
print("n=5时，手动knn正确率为：",my_corr[2])
print("n=5时，调包knn正确率为：",knn_corr[2])

name_list=['k=1','k=3','k=5']

x = list(range(len(my_corr)))
total_width, n = 0.4, 2
width = total_width / n
plt.ylim([0,1.1])
plt.bar(x, my_corr, width=width, label='my_KNN', fc='b')
for a,b in zip(x,my_corr):
    plt.text(a,b,'%.3f'%b,ha='center',va='bottom',fontsize=10)
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, knn_corr, width=width, label='sk_KNN', tick_label=name_list, fc='r')

for a,b in zip(x,knn_corr):
    plt.text(a,b,'%.3f'%b,ha='center',va='bottom',fontsize=10)
plt.title("不同k值下两种knn的预测准确率对比柱状图")
plt.xlabel("K值",fontsize=10)
plt.ylabel("准确率",fontsize=10)
plt.legend()
plt.show()


















