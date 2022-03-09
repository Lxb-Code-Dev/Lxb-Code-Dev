import torch
from torch import nn
import pandas as pd
import scanpy as sc
import numpy as np
import copy
import torch.utils.data as Data
from loss import loss

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


# def loss(x,pi,u,theta):
#     eps=torch.tensor(1e-10)
#     #事实上u即为y_pred，即补差的值
#     #注意是负对数，因此我们可以将乘法和除法变为加减法，
#     theta=torch.minimum(theta,torch.tensor(1e6))
#     t1 = torch.lgamma(theta + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + theta + eps)
#     t2 = (theta + x) * torch.log(1.0 + (u / (theta + eps))) + (x * (torch.log(theta + eps) - torch.log(u + eps)))
#     nb=t1+t2
#     nb = torch.where(torch.isnan(nb), torch.zeros_like(nb) + np.inf, nb)
#     nb_case=nb- torch.log(1.0-pi+eps)
#     zero_nb = torch.pow(theta / (theta + u + eps), theta)
#     zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
#     res = torch.where(torch.less(x, 1e-8), zero_case, nb_case)
#     res = torch.where(torch.isnan(res), torch.zeros_like(res) + np.inf, res)
#     return torch.mean(res)

if __name__=="__main__":
    pass
    # 定义超参数
    # batch_size = 32
    # learning_rate = 0.0003
    # num_epochs = 5  # 训练次数
    # # 判断GPU是否可用
    # use_gpu = torch.cuda.is_available()
    # raw_data = sc.read_csv("data.csv")
    # adata = raw_data.copy()
    # sc.pp.normalize_total(adata, inplace=True)
    # sc.pp.log1p(adata)
    # sc.pp.scale(adata)
    # adata = torch.tensor(adata.X.T)
    # raw_data = torch.tensor(raw_data.X.T)
    # dataset = Data.TensorDataset(adata, raw_data)
    # batch_train_Data = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    #
    # dca_model=dca(adata.size()[1])
    # if use_gpu:
    #     dca_model=dca_model.cuda()
    # optimizer = torch.optim.RMSprop(dca_model.parameters(), lr=learning_rate)
    # criterion = loss()  # 损失函数
    # #开始模型训练
    # for epoch in range(50):
    #     print('*' * 10)
    #     print(f'epoch {epoch + 1}')
    #     running_loss = 0.0  # 初始值
    #     # input_x=adata
    #     # true_data=raw_data
    #     for step, (input, raw) in enumerate(batch_train_Data):
    #         if use_gpu:
    #             input=input.cuda()
    #             #raw=raw.cuda()
    #         pi,u,theta=dca_model(input)
    #         loss_ = criterion(input, pi, u, theta)
    #         running_loss += loss_.item()
    #         optimizer.zero_grad()  # 梯度归零
    #         loss_.backward()  # 后向传播
    #         optimizer.step()  # 更新参数
    #     print(f'Finish {epoch + 1} epoch, Loss: {running_loss :.6f}')
    #
    #     # 向前传播
    #     # for i in range(10):
    #     #     input_x=adata[epoch*10+i]
    #     #     true_data=raw_data[epoch*10+i]
    #     #     pi,u,theta = dca_model(input_x)  # 前向传播
    #     #     loss_ = loss(true_data,pi,u,theta)  # 计算loss
    #     #     running_loss += loss_.item()  # loss求和
    #     #     # 向后传播
    #     #     optimizer.zero_grad()  # 梯度归零
    #     #     loss_.backward()  # 后向传播
    #     #     optimizer.step()  # 更新参数
    #     # print(f'Finish {epoch + 1} epoch, Loss: {running_loss/10:.6f}')

