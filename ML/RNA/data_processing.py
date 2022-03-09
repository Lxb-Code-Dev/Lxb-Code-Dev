import scanpy as sc
import copy
import torch
import torch.utils.data as Data
import pandas as pd
import numpy as np

def get_adata(path,path2=None,batch_size=32):
    if path2 is None:
        test_data = torch.tensor(np.array(pd.read_csv(path)).T,dtype=torch.float)
        # test_data_sum = torch.sum(test_data, 1)  # 求和每个基因的表达量
        # test_data_mid = torch.median(test_data, 1).values  # 中位数
        # test_data_s = torch.div(test_data_sum, test_data_mid)
        # test_data_diag = torch.diag(test_data_s)
        # test_data_inver = torch.inverse(test_data_diag)
        # test_data_mul = torch.log(test_data_inver * test_data + 1)
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
    # adata = sc.read_csv(path)
    # sc.pp.normalize_total(adata, inplace=True)
    # sc.pp.log1p(adata)
    # sc.pp.scale(adata)
    # adata = torch.tensor(adata.X.T)
    # raw_data = torch.tensor(raw_data.X.T)
    # dataset = Data.TensorDataset(adata, raw_data)
    # batch_train_Data = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    # return batch_train_Data,adata.size()[1]

# test_truedata=torch.tensor(np.array(pd.read_csv("test_truedata.csv")).T,dtype=torch.float)
# test_data=torch.tensor(np.array(pd.read_csv("test_data.csv")).T,dtype=torch.float)
# test_data_sum=torch.sum(test_data,1)  #求和每个基因的表达量
# test_data_mid=torch.median(test_data,1).values  #中位数
# test_data_s=torch.div(test_data_sum,test_data_mid)
# test_data_diag=torch.diag(test_data_s)
# test_data_inver=torch.inverse(test_data_diag)
# test_data_mul=torch.log(torch.matmul(test_data_inver,test_data)+1)
# print(test_data_mul)

