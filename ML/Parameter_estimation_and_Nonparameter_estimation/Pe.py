import numpy as np
import math
import matplotlib.pyplot as plt
def data_generation(n1,n2,n3):
    x1=np.random.multivariate_normal((1,1),[[2,0],[0,2]],n1)
    x1_0=[[0]]*n1
    x2=np.random.multivariate_normal((4,4),[[2,0],[0,2]],n2)
    x2_1=[[1]]*n2
    x3=np.random.multivariate_normal((8,1),[[2,0],[0,2]],n3)
    x3_2=[[2]]*n3
    re1=np.r_[x1,x2,x3]
    re=np.c_[re1,x1_0+x2_1+x3_2]
    return re
X1=data_generation(333,333,334)
X2=data_generation(600,300,100)
plt.subplot(1,2,1)
plt.scatter([x[0] for x in X1[0:333,::]],[x[1] for x in X1[0:333,::]],c='r')
plt.scatter([x[0] for x in X1[333:666,::]],[x[1] for x in X1[333:666,::]],c='g')
plt.scatter([x[0] for x in X1[666:,::]],[x[1] for x in X1[666:,::]],c='b')
plt.subplot(1,2,2)
plt.scatter([x[0] for x in X2[0:600,::]],[x[1] for x in X2[0:600,::]],c='r')
plt.scatter([x[0] for x in X2[600:900,::]],[x[1] for x in X2[600:900,::]],c='g')
plt.scatter([x[0] for x in X2[900:,::]],[x[1] for x in X2[900:,::]],c='b')
plt.show()

class Pe():
    def __init__(self,dataset,p1,p2,p3):
        self.data=dataset
        self.p1=p1
        self.p2=p2
        self.p3=p3

    def f(self,u1,u2,o,x,y):
        '''
        求对应点的概率密度
        :param u1: 均值1
        :param u2: 均值2
        :param o: 方差
        :param x: x
        :param y: y
        :return: (x,y)的概率密度
        '''
        return math.exp((-1/2)*((x-u1)**2/o+(y-u2)**2/o))/(2*o*math.pi)

    def classify(self):
        ff=[0,0,0]
        score=0
        for i,xy in enumerate(self.data):
            ff[0]=self.f(1,1,2,xy[0],xy[1])*self.p1
            ff[1]=self.f(4,4,2,xy[0],xy[1])*self.p2
            ff[2]=self.f(8,1,2,xy[0],xy[1])*self.p3
            max_f_index=ff.index(max(ff))
            if max_f_index==xy[2]:
                score+=1
                continue
        return 1-score/1000
    def p(self,h,x,y,Xn):
        '''

        :param h: h
        :param N: 该分类元素个数
        :param x : x
        :param y :y
        :return: 返回p
        '''
        total=0.0
        for i in Xn:
            total+=math.exp(-((x-i[0])**2+(y-i[1])**2)/(2*h*h))/math.sqrt(2*h*h*math.pi)
        return total/len(Xn)

    def gauss_ker(self):
        data0=self.data[:self.p1,::]
        data1=self.data[self.p1:self.p1+self.p2,::]
        data2=self.data[self.p1+self.p2:,::]
        total_score=[]
        for h in [0.1,0.5,1,1.5,2]:
            ff = [0, 0, 0]
            score = 0
            for i,xy in enumerate(self.data):
                ff[0]=self.p(h,xy[0],xy[1],data0)
                ff[1]=self.p(h,xy[0],xy[1],data1)
                ff[2]=self.p(h,xy[0],xy[1],data2)
                max_f_index = ff.index(max(ff))
                if max_f_index == xy[2]:
                    score += 1
                    continue
            total_score.append( 1-score / (self.p2+self.p1+self.p3))
        return total_score




pe1=Pe(X1,333,333,334)
pe2=Pe(X2,600,300,100)
print("数据集X1分类错误率为：",pe1.classify())
print("数据集X1核函数分类错误率为：",pe1.gauss_ker())
print("数据集X2分类错误率为：",pe2.classify())
print("数据集X2核函数分类错误率为：",pe2.gauss_ker())



