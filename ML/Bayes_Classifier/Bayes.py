'''
从wine.names中获得的信息如下
wine.data数据集中：
种类：三种类型，数目分别为59  71  48，共178条数据
属性：13种化学成分，都是连续数值类型
wine.data的第一列数据为分类数据，三种类型1 2 3
'''
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
data = np.array(pd.read_csv('wine.data', header=None).values.tolist())

class Bayes():
    def __init__(self,data):
        self.data=data
        self.train_data=None  #41 49 33
        self.train_target=None
        self.test_data=None
        self.test_target=None
        self.p=None
    def data_processing(self,p=0.3):
        '''
        按照分层采样的方法对数据集按比例p进行划分，返回划分后的训练集和测试集
        :param data: 待处理的数据集
        :param p: 测试集占比
        :return:
        '''
        self.p=p
        data_class1 = self.data[:59,::]
        data_class2 = self.data[59:130,::]
        data_class3 = self.data[130:,::]
        #首先对每种类型数据进行打乱
        np.random.shuffle(data_class1)
        np.random.shuffle(data_class2)
        np.random.shuffle(data_class3)
        self.train_data=np.r_[data_class1[0:int((1-p)*59),::],data_class2[0:int((1-p)*71),::],data_class3[0:int((1-p)*48),::]][::,1:]
        self.train_target=np.r_[data_class1[0:int((1-p)*59),::],data_class2[0:int((1-p)*71),::],data_class3[0:int((1-p)*48),::]][::,:1]
        self.test_data=np.r_[data_class1[int((1-p)*59):,::],data_class2[int((1-p)*71):,::],data_class3[int((1-p)*48):,::]][::,1:]
        self.test_target=np.r_[data_class1[int((1-p)*59):,::],data_class2[int((1-p)*71):,::],data_class3[int((1-p)*48):,::]][::,:1]
        #数据标准化，注意测试集标准化所用期望和标准差与训练集相同
        means,stds=self.prior_para(self.train_data)
        self.train_data=(self.train_data-means)/stds
        self.test_data=(self.test_data-means)/stds
    def prior_para(self,train_data):
        '''

        :param train_data: 训练集
        :return: 在同一类型下，返回每种属性对应的期望和标准差
        '''
        means=np.mean(train_data,axis=0)
        stds=np.std(train_data,axis=0)
        return means,stds
    def Gauss(self,mean,std,x):
        '''

        :param mean: 高斯分布的期望
        :param std: 高斯分布的标准差
        :param x: x
        :return: x对应的高斯概率密度
        '''
        return math.exp(-((x-mean)**2)/(2*std**2))/(math.sqrt(2*math.pi)*std)
    def classify(self):
        '''

        :param test: 测试集
        :return: 分类结果
        '''
        class_re=[]
        mean1,std1=self.prior_para(self.train_data[:int((1-self.p)*59),::])
        mean2,std2=self.prior_para(self.train_data[int((1-self.p)*59):int((1-self.p)*71)+int((1-self.p)*59),::])
        mean3,std3=self.prior_para(self.train_data[int((1-self.p)*71)+int((1-self.p)*59):,::])
        for i in self.test_data:
            p1=int((1-self.p)*59)/123
            p2=int((1-self.p)*71)/123
            p3=int((1-self.p)*48)/123
            for j,z in enumerate(i):
                p1*=self.Gauss(mean1[j],std1[j],z)
                p2*=self.Gauss(mean2[j],std2[j],z)
                p3*=self.Gauss(mean3[j],std3[j],z)
            class_re.append([p1,p2,p3].index(max([p1,p2,p3]))+1)
        return class_re

    def run(self,p):
        '''
        分类器整体流程执行
        :return:
        '''
        self.data_processing(p)
        predict=self.classify()
        score=list(self.test_target.T[0]-predict).count(0)/len(predict)
        print('======================分割线======================')
        print('测试集占比为'+str(p)+'时，分类准确率为：',score)
        num1_1=predict[:59-(int((1-self.p)*59))].count(1)
        num1_2=predict[:59-(int((1-self.p)*59))].count(2)
        num1_3=predict[:59-(int((1-self.p)*59))].count(3)
        num2_1=predict[59-(int((1-self.p)*59)):59+71-(int((1-self.p)*71))-int((1-self.p)*59)].count(1)
        num2_2=predict[59-(int((1-self.p)*59)):59+71-(int((1-self.p)*71))-int((1-self.p)*59)].count(2)
        num2_3=predict[59-(int((1-self.p)*59)):59+71-(int((1-self.p)*71))-int((1-self.p)*59)].count(3)
        num3_1=predict[59+71-(int((1-self.p)*71))-int((1-self.p)*59):].count(1)
        num3_2 = predict[59 + 71 - (int((1 - self.p) * 71)) - int((1 - self.p) * 59):].count(2)
        num3_3 = predict[59 + 71 - (int((1 - self.p) * 71)) - int((1 - self.p) * 59):].count(3)
        confusion_mat=np.array([[num1_1,num1_2,num1_3],[num2_1,num2_2,num2_3],[num3_1,num3_2,num3_3]])
        col=['预测为种类1','预测为种类2','预测为种类3']
        row=['实际为种类1','实际为种类2','实际为种类3']
        plt.title("测试集占比为"+str(p)+"时的混淆矩阵")
        plt.table(cellText=confusion_mat,
                        colLabels=col,
                        rowLabels=row,
                        loc='center',
                        cellLoc='center',
                        rowLoc='center')

        plt.axis('off')
        plt.show()
        print('种类1精度为：',num1_1/sum(confusion_mat[0]))
        print('种类2精度为：', num2_2 / sum(confusion_mat[1]))
        print('种类3精度为：', num3_3 / sum(confusion_mat[2]))

        print('种类1召回率为：', num1_1 / sum(confusion_mat.T[0]))
        print('种类2召回率为：', num2_2 / sum(confusion_mat.T[1]))
        print('种类3召回率为：', num3_3 / sum(confusion_mat.T[2]))

        print('种类1 F值为：', 2/(sum(confusion_mat[0])/num1_1+ sum(confusion_mat.T[0])/num1_1 ))
        print('种类2 F值为：', 2/(sum(confusion_mat[1])/num2_2+ sum(confusion_mat.T[1])/num2_2 ))
        print('种类3 F值为：', 2/(sum(confusion_mat[2])/num3_3+ sum(confusion_mat.T[2])/num3_3 ))
bayes=Bayes(data)
bayes.run(0.2)
bayes.run(0.3)
bayes.run(0.4)
