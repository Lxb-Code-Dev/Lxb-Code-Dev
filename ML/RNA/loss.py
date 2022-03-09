import torch
import numpy as np
from torch import nn

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