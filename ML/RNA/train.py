from DCA import dca
import torch
import scanpy as sc
import torch.utils.data as Data
import numpy as np
import copy
from loss import loss
from data_processing import get_adata
import math

# 定义超参数
learning_rate = [0.0001,0.0002,0.0003,0.001,0.002,0.003]
epochs=[3000]
# num_epochs = 300 # 训练次数
# param_dict=dict()
# 判断GPU是否可用
use_gpu = torch.cuda.is_available()
# batch_train_Data,data_size=get_adata("data.csv",batch_size)
#实例化模型
class Train():
    def __init__(self,batch_size,learning_rate,epochs):
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.running_loss=0.0
    def run_train(self):
        batch_train_Data, data_size = get_adata(path="test_data.csv",path2="test_truedata.csv", batch_size=self.batch_size)
        dca_model=dca()
        if use_gpu:
            dca_model=dca_model.cuda()
        criterion = loss()  # 损失函数
        optimizer = torch.optim.RMSprop(dca_model.parameters(), lr=self.learning_rate)
        ciru=None
        #开始模型训练
        dds=(5000,0,0)
        for epoch in range(self.epochs):
            # print('*' * 10)
            # print(f'epoch {epoch + 1}')
            self.running_loss = 0.0  # 初始值
            # input_x=adata
            # true_data=raw_data
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
            print("epochs:",epoch+1," learning_rate:",self.learning_rate," loss:",self.running_loss/ciru)

            distance = torch.tensor(0, dtype=torch.float)
            distance = distance.cuda()
            for step, (input, raw) in enumerate(batch_train_Data):
                ciru += 1
                if use_gpu:
                    input = input.cuda()
                    raw = raw.cuda()
                pi, u, theta, u_loss = dca_model(input)
                b = torch.where(input != 0, input, torch.ceil(u)) #补差后
                c = torch.where(input == 0, 1, 0)  #补差前
                e = torch.where(raw == 0, 1, 0)      #真实数据
                f = torch.where(b == 0, 1, 0)
                cc = torch.where(e==1,0,c)   #补差前的dropout分布
                bb = torch.where(e==1,0,f) #补差后的dropout分布
                print("初始dropout率：",torch.sum(cc)/cc.numel()," 补差后的dropout率：",torch.sum(bb)/bb.numel())
                distance += torch.dist(b, raw, p=2)
            if dds[0]>distance:
                dds=(distance,self.learning_rate,epoch+1)
            print("欧氏距离为：", distance)
        return dds
        # batch_train_Data, data_size = get_adata("test_data.csv", "test_truedata.csv", 200)
        # distance = torch.tensor(0,dtype=torch.float)
        # distance=distance.cuda()
        # for step, (input, raw) in enumerate(batch_train_Data):
        #     ciru += 1
        #     if use_gpu:
        #         input = input.cuda()
        #         raw=raw.cuda()
        #     pi, u, theta,u_loss = dca_model(input)
        #     b=torch.where(input!=0,input,torch.round(u))
        #     distance += torch.dist(b, raw, p=2)
        # print("欧氏距离为：", distance)
        # if not math.isinf(self.running_loss/ciru):
        #     torch.save(dca_model,"./model/epoch_"+str(self.epochs)+"_lr_"+str(self.learning_rate).split('.')[1]+".pt")
        #return {"epochs":self.epochs,"learning_rate":self.learning_rate,"loss":self.running_loss/ciru}
result=[]
for epoch in epochs:
    for rate in learning_rate:
        train_fun = Train(200,rate,epoch)
        result.append(train_fun.run_train())
print(result)
#losses=[re["loss"] for re in result]
#print("训练损失值最小为：",min(losses))
# print("最优超参数为：",result[losses.index(min(losses))])