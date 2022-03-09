import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
class HC():
    def __init__(self,n,m,var):
        '''

        :param n: 样本数
        :param m: 标签数
        '''
        self.var=var
        self.n=n
        self.m=m
        self.data=None
        self.dis_mat=None
        self.dis_temp=None
        self.raw=None
        self.hc=[[i] for i in range(n)]
    def data_generation(self):
        '''

        :return: 样本以及样本举例矩阵
        '''
        mid=[]
        for i in range(1,self.m+1):
            x=np.random.multivariate_normal((8*i,8*i,8*i),[[self.var,0,0],[0,self.var,0],[0,0,self.var]],int(self.n/self.m))
            y=[[i]]*(int(self.n/self.m))
            z=np.c_[x, y]
            mid.append(z)
        self.raw=mid.copy()
        re=mid.pop(0)
        for i in mid:
            re=np.r_[re,i]
        mat=[]
        for i in re:
            temp=[]
            for j in re:
                temp.append(math.sqrt(sum([(p-q)**2 for p,q in zip(i[:-1],j[:-1])])))
            mat.append(temp)
        self.data=re
        self.dis_mat=np.array(mat)
        #刚生成时，n个类别，因此类别距离矩阵与n维距离矩阵相同
        self.dis_temp=self.dis_mat

    def dis_SI(self):
        '''
        最短距离
        :return: None
        '''
        size=len(self.hc)
        dis_tem=np.zeros((size,size))
        for inx,i in enumerate(self.hc):
            for iny,j in enumerate(self.hc):
                dis_tem[inx][iny]=min([self.dis_mat[m][n] for m in i for n in j])
        self.dis_temp=dis_tem
    def dis_CI(self):
        '''
        最长距离
        :return:
        '''
        size=len(self.hc)
        dis_tem=np.zeros((size,size))
        for inx,i in enumerate(self.hc):
            for iny,j in enumerate(self.hc):
                dis_tem[inx][iny]=max([self.dis_mat[m][n] for m in i for n in j])
        self.dis_temp=dis_tem
    def dis_AI(self):
        '''
        平均距离
        :return:
        '''
        size=len(self.hc)
        dis_tem=np.zeros((size,size))
        for inx,i in enumerate(self.hc):
            for iny,j in enumerate(self.hc):
                dis_tem[inx][iny]=sum([self.dis_mat[m][n] for m in i for n in j])/(len(i)*len(j))
        self.dis_temp=dis_tem
    def SI(self):
        '''

        :return: None
        '''
        #方法不同，因此先初始化
        self.hc = [[i] for i in range(self.n)]
        self.dis_temp = self.dis_mat
        while True:
            # 首先考虑将距离矩阵对角线上的值设定为一个很大的值，
            # 因为在本算法中我们要找最小值，因此我们将对角线上的0改为当前矩阵的最大值+1
            row, col = np.diag_indices_from(self.dis_temp)
            self.dis_temp[row, col] = self.dis_temp.max() + 1
            idx, idy = self.dis_temp.shape
            index = int(self.dis_temp.argmin())
            x = int(index / idy)
            y = index % idy
            self.hc[x].extend(self.hc[y])
            self.hc.pop(y)
            if len(self.hc)==self.m:
                return
            self.dis_SI()
    def CI(self):
        '''

        :return: None
        '''
        # 方法不同，因此先初始化
        self.hc = [[i] for i in range(self.n)]
        self.dis_temp = self.dis_mat
        while True:
            # 首先考虑将距离矩阵对角线上的值设定为一个很大的值，
            # 因为在本算法中我们要找最小值，因此我们将对角线上的0改为当前矩阵的最大值+1
            row, col = np.diag_indices_from(self.dis_temp)
            self.dis_temp[row, col] = self.dis_temp.max() + 1
            idx, idy = self.dis_temp.shape
            index = int(self.dis_temp.argmin())
            x = int(index / idy)
            y = index % idy
            self.hc[x].extend(self.hc[y])
            self.hc.pop(y)
            if len(self.hc) == self.m:
                return
            self.dis_CI()
    def AI(self):
        '''

        :return: None
        '''
        # 方法不同，因此先初始化
        self.hc = [[i] for i in range(self.n)]
        self.dis_temp = self.dis_mat
        while True:
            # 首先考虑将距离矩阵对角线上的值设定为一个很大的值，
            # 因为在本算法中我们要找最小值，因此我们将对角线上的0改为当前矩阵的最大值+1
            row, col = np.diag_indices_from(self.dis_temp)
            self.dis_temp[row, col] = self.dis_temp.max() + 1
            idx, idy = self.dis_temp.shape
            index = int(self.dis_temp.argmin())
            x = int(index / idy)
            y = index % idy
            self.hc[x].extend(self.hc[y])
            self.hc.pop(y)
            if len(self.hc) == self.m:
                return
            self.dis_AI()

    def run(self):
        '''

        :return: 未聚类结果re以及三种算法的聚类结果re1 re2 re3
        '''
        self.data_generation() #执行数据生成器
        self.SI()
        re1=[[self.data[i] for i in self.hc[j]] for j in range(self.m)]
        self.CI()
        re2=[[self.data[i] for i in self.hc[j]] for j in range(self.m)]
        self.AI()
        re3=[[self.data[i] for i in self.hc[j]] for j in range(self.m)]
        return self.raw,re1,re2,re3
hc=HC(2000,5,4)
raw,re1,re2,re3=hc.run()
draw=[raw,re1,re2,re3]
fig = plt.figure()
ax = [221, 222, 223, 224]
title=['raw','SI','CI','AI']

hc2=HC(2000,5,6)
raw1,re11,re21,re31=hc2.run()
draw1=[raw1,re11,re21,re31]

for index in range(4):
    ax[index] = fig.add_subplot(ax[index], projection='3d')
    ax[index].set_title(title[index])
    color=['b','c','g','r','k']
    for i in range(5):
        ax[index].scatter3D([x[0] for x in list(draw[index][i])], [x[1] for x in list(draw[index][i])], [x[2] for x in list(draw[index][i])], c=color[i])
plt.show()

fig1 = plt.figure()
ax1 = [221, 222, 223, 224]
title1=['raw','SI','CI','AI']
for index in range(4):
    ax1[index] = fig1.add_subplot(ax1[index], projection='3d')
    ax1[index].set_title(title1[index])
    color=['b','c','g','r','k']
    for i in range(5):
        ax1[index].scatter3D([x[0] for x in list(draw1[index][i])], [x[1] for x in list(draw1[index][i])], [x[2] for x in list(draw1[index][i])], c=color[i])
plt.show()



