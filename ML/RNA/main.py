#from DCA import dca
import scanpy as sc
import numpy as np
import copy
#from loss import loss
#from data_processing import get_adata
import math
import pandas as pd
import torch
from torch import nn
import torch.utils.data as Data

class dca(nn.Module):
    def __init__(self,indim=1000,encode_dim=64,bottleneck_dim=32,decode_dim=64):
        super(dca, self).__init__()
        self.batchnorm = nn.Sequential(nn.BatchNorm1d(indim))
        self.encode = nn.Sequential(nn.Linear(in_features=indim, out_features=encode_dim)
                                    ,nn.ReLU(True))
        self.batchnorm1 = nn.Sequential(nn.BatchNorm1d(encode_dim))
        self.bottleneck = nn.Sequential(nn.Linear(in_features=encode_dim,out_features=bottleneck_dim)
                                    ,nn.ReLU(True))
        self.batchnorm2 = nn.Sequential(nn.BatchNorm1d(bottleneck_dim))
        self.decode = nn.Sequential(nn.Linear(in_features=bottleneck_dim,out_features=decode_dim)
                                    ,nn.ReLU(True))
        self.output1 = nn.Sequential(nn.Linear(in_features=decode_dim,out_features=indim)
                                    )
        self.output2 = nn.Sequential(nn.Linear(in_features=decode_dim, out_features=indim)
                                     )
        self.output3 = nn.Sequential(nn.Linear(in_features=decode_dim, out_features=indim)
                                     )
    def forward(self, x):
        test_data_sum = torch.sum(x, 1)  # 求和每个基因的表达量
        test_data_mid = torch.median(test_data_sum)  # 中位数
        test_data_s = test_data_sum/test_data_mid
        test_data_diag = torch.diag(test_data_s)
        test_data_inver = torch.inverse(test_data_diag)
        test_data_mul = torch.log(torch.matmul(test_data_inver,x)+1)
        #test_data_mean = torch.mean(test_data_mul, 1)
        #test_data_std = torch.std(test_data_mul, 1)
        #adata = (test_data_mul - test_data_mean) / test_data_std
        x=self.batchnorm(test_data_mul)
        x=self.encode(x)
        x=self.batchnorm1(x)
        x=self.bottleneck(x)
        x=self.batchnorm2(x)
        x=self.decode(x)
        M=torch.exp(self.output1(x))
        M_loss=torch.matmul(test_data_diag,M)
        pai=torch.sigmoid(self.output2(x))
        theta=torch.exp(self.output3(x))
        return pai,M,theta,M_loss

def get_adata(path,path2=None,batch_size=32):
    if path2 is None:
        test_data = torch.tensor(np.array(pd.read_csv(path)).T,dtype=torch.float)
        # test_data_sum = torch.sum(test_data, 1)  # 求和每个基因的表达量
        # test_data_mid = torch.median(test_data, 1).values  # 中位数
        # test_data_s = torch.div(test_data_sum, test_data_mid)
        # test_data_diag = torch.diag(test_data_s)
        # test_data_inver = torch.inverse(test_data_diag)
        # test_data_mul = torch.log(test_data_inver * test_data + 1)
        # test_data_mean=torch.mean(test_data_mul,1)
        # test_data_std=torch.std(test_data_mul,1)
        # adata=(test_data_mul-test_data_mean)/test_data_std
        dataset = Data.TensorDataset(test_data, test_data)   #处理后的数据以及对角矩阵
        batch_train_Data = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        return batch_train_Data, test_data.shape[0]
    else:
        test_data = torch.tensor(np.array(pd.read_csv(path)).T,dtype=torch.float)
        test_truedata = torch.tensor(np.array(pd.read_csv(path2)).T,dtype=torch.float)
        # raw_data = sc.read_csv(path2)
        # adata = sc.read_csv(path)
        # adata = torch.tensor(adata.X.T)
        # raw_data = torch.tensor(raw_data.X.T)
        dataset = Data.TensorDataset(test_data, test_truedata)
        batch_train_Data = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        return batch_train_Data, test_data.shape[0]

class loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x,pi,u,theta):
        eps = torch.tensor(1e-10)
        # 事实上u即为y_pred，即补差的值
        # 注意是负对数，因此我们可以将乘法和除法变为加减法，
        theta = torch.minimum(theta, torch.tensor(1e6))
        t1 = torch.lgamma(theta + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + theta + eps)
        t2 = (theta + x) * torch.log(1.0 + (u / (theta + eps))) + (x * (torch.log(theta + eps) - torch.log(u + eps)))
        nb = t1 + t2
        nb = torch.where(torch.isnan(nb), torch.zeros_like(nb) + np.inf, nb)
        nb_case = nb - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(theta / (theta + u + eps), theta)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        res = torch.where(torch.less(x, 1e-8), zero_case, nb_case)
        res = torch.where(torch.isnan(res), torch.zeros_like(res) + np.inf, res)
        return torch.mean(res)
# 定义超参数
learning_rate = 0.0001
epochs=2000
use_gpu = torch.cuda.is_available()
class Predict():
    def __init__(self,batch_size,learning_rate,epochs,a):
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.running_loss=0.0
        self.a=a
    def run_predict(self):
        batch_train_Data, data_size = get_adata(path="data.csv", batch_size=self.batch_size)
        dca_model=None
        if self.a==0:
            dca_model=dca()
            if use_gpu:
                dca_model=dca_model.cuda()
            criterion = loss()
            optimizer = torch.optim.RMSprop(dca_model.parameters(), lr=self.learning_rate)
            ciru=None
            #开始模型训练
            print("根据超参数开始训练")
            for epoch in range(self.epochs):
                self.running_loss = 0.0
                ciru=0
                for step, (input, raw) in enumerate(batch_train_Data):
                    ciru+=1
                    if use_gpu:
                        input=input.cuda()
                        #raw=raw.cuda()
                    pi,u,theta,u_loss=dca_model(input)
                    loss_ = criterion(input, pi, u_loss, theta)
                    self.running_loss += loss_.item()
                    optimizer.zero_grad()  # 梯度归零
                    loss_.backward()  # 后向传播
                    optimizer.step()  # 更新参数
                #print(f'Finish {epoch + 1} epoch, Loss: {self.running_loss :.6f}')
                print("epochs:",epoch+1," loss:",self.running_loss/ciru)
            #保存模型
            torch.save(dca_model,"data_predict_model.pt")
            print("模型已保存到当前目录下：data_predict_model.pt")
        elif self.a==1:
            dca_model = torch.load('data_predict_model.pt')
        #进行补差
        for step, (input, raw) in enumerate(batch_train_Data):
            if use_gpu:
                input = input.cuda()
                #raw = raw.cuda()
            pi, u, theta, u_loss = dca_model(input)
            b = torch.where(input != 0, input, torch.round(u))
            c = torch.where(input==0,1,0)
            d=torch.where(b==0,1,0)
            dropout_num_raw=torch.sum(c)/c.numel()
            dropout_num_predict=torch.sum(d)/d.numel()
            print("初始0的比率（注意不是dropout率，因为没有真实数据，因此这里用0的占比逼近）：",dropout_num_raw," 补差后的0的比率：",dropout_num_predict)
            predict_data=b.cpu().detach().numpy().T
            head=['Cell'+str(i) for i in range(1,predict_data.shape[1]+1)]
            np.savetxt("data_predict.csv",predict_data,delimiter = ',',fmt='%d',header=','.join(head),comments='')
            print("补差结果已保存到当前目录下的data_predict.csv文件中")

data = torch.tensor(np.array(pd.read_csv("data.csv")).T, dtype=torch.float)
batch_predict_Data, data_size = get_adata(path="data.csv", batch_size=data.shape[0])
print("如果选择已有模型（用data数据集训练的）进行补差，请输入”1“；如果想要重新根据data数据集进行训练并做补差，请输入”0“；重新训练可能需要等待一段时间")
a=eval(input("请选择补差方式："))
Predict_FUN=Predict(data_size,learning_rate,epochs,a)
Predict_FUN.run_predict()





