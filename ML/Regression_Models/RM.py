import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data_a=pd.read_csv("dataset_regression.csv",usecols=[1,2])
data_a=data_a.values.tolist()
data_b=pd.read_csv("winequality-white.csv").values.tolist()
data_b=np.array(data_b)

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False


def ols(data):
    '''
    实现最小二乘法
    :param data: 数据集
    :return: 最小二乘解
    '''
    ave_x=sum([i[0] for i in data])/len(data)
    ave_y=sum([i[-1] for i in data])/len(data)
    sum_xy=sum([i[0]*i[-1] for i in data])
    sum_xx=sum([ i[0]**2 for i in data])
    param_1=(sum_xy - ave_x * ave_y * len(data)) / (sum_xx - len(data) * ave_x * ave_x)
    param_0=ave_y-param_1*ave_x
    return param_0,param_1

def mse(data):
    '''

    :param data: 数据集
    :return: 均方误差
    '''
    param0,param1=ols(data)
    y0=[param0+param1*i[0] for i in data]
    y1=[i[-1] for i in data]
    return sum([(i-j)**2 for i,j in zip(y0,y1)])/len(data)


def data_split(data,p):
    '''
    分割数据集
    :param p: 训练集占比
    :param data: 完整数据集
    :return: 训练集和测试集
    '''
    np.random.shuffle(data)
    return data[:int(data.shape[0]*p),::],data[int(data.shape[0]*p):,::]

def data_standard(data,mean=[],stand=[]):
    '''

    :param data: 数据集
    :return: 标准化后的数据集
    '''
    data_x=data[::,:-1].T
    if not len(mean):
        mean=data_x.mean(axis=1)
    if not len(stand):
        stand=data_x.std(axis=1)
    for i in range(len(data_x)):
        data_x[i]=(data_x[i]-mean[i])/stand[i]
    return data_x.T,mean,stand




def bgd(train_data,test_data,init_w,lr,iter_num):
    '''
    批量梯度下降
    :param train_data:训练数据集
    :param init_w:初始化权重矩阵
    :param lr:学习率
    :param iter_num:迭代次数
    :return:迭代后的权重矩阵
    '''
    train_data_x,mean,stand=data_standard(train_data)
    one=np.ones(len(train_data_x))
    train_data_x=np.c_[one,train_data_x]
    train_data_y=train_data[::,-1:]
    train_mse=[]
    test_data_x,mean,stand = data_standard(test_data,mean,stand)
    one = np.ones(len(test_data_x))
    test_data_x = np.c_[one, test_data_x]
    test_data_y = test_data[::, -1:]
    test_mse = []

    for i in range(iter_num):
        train_predict_y=train_data_x.dot(init_w)
        test_predict_y = test_data_x.dot(init_w)
        test_mse.append(
            np.dot(test_data_y.T[0] - test_predict_y.T[0], test_data_y.T[0] - test_predict_y.T[0]) / len(
                test_data_x))
        train_mse.append(
            np.dot(train_data_y.T[0] - train_predict_y.T[0], train_data_y.T[0] - train_predict_y.T[0]) / len(
                train_data_x))
        train_grad=(train_data_x.T.dot(train_data_y-train_predict_y))/len(train_data_x)*lr
        # for j in range(len(init_w)):
        #     init_w[j][0]=init_w[j][0]+(lr/len(train_data_x)*((train_data_y-train_data_x.dot(init_w)).T.dot(np.array([[z] for z in train_data_x.T[j]]))))
        init_w=init_w+train_grad


    return init_w,train_mse,test_mse



print("==============================基本要求a)==========================================")
x=[i[-2] for i in data_a]
y=[i[-1] for i in data_a]
param_0,param_1=ols(data_a)
Y=[i*param_1+param_0 for i in x]
plt.xlabel(u'X')
plt.ylabel(u'Y')
plt.title('散点图与回归曲线')
plt.scatter(x,y)
plt.plot(x, Y)
plt.show()
print('训练集MSE为：',mse(data_a))
print("==============================基本要求b)及中级要求==================================")
init_w=np.array([[float(i-i+1)] for i in range(data_b.shape[1])])
train_data,test_data=data_split(data_b,0.8)


for z,i in enumerate([0.001,0.003,0.1,0.3]):
    W,train_mse,test_mse=bgd(train_data,test_data,init_w,i,1000)
    print("学习率为"+str(i)+"时训练集最终MSE:",train_mse[-1])
    print("学习率为"+str(i)+"时测试集最终MSE:", test_mse[-1])
    plt.subplot(2,2,z+1)
    plt.plot([j for j in range(1000)], train_mse, 'r-', label=u'Train_mse')
    plt.plot([j for j in range(1000)], test_mse, 'b-', label=u'Test_mse')
    plt.legend()
    plt.xlabel(u'iters')
    plt.ylabel(u'loss')
    plt.title('学习率lr为'+str(i)+'时训练集与测试集的MSE收敛曲线图像')
plt.show()