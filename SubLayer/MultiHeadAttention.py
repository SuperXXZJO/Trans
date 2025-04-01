import torch
import torch.nn as nn
import torch.nn.functional as Func
import math

class MultiheadAttention(nn.Module):
    def __init__(self,heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads #
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # keep output dim same with input
        self.out = nn.Linear(d_model, d_model)

    def attention(self,q,k,v,d_k,mask = None, dropout = None):
        # attention score
        scores = torch.matmul(q,k.transpose(-2,-1)) / math.sqrt(d_k)

        # maskenabled 
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask==0,-1e9)

        scores = Func.softmax(scores, dim = -1)
        if dropout is not None:
            scores = dropout(scores)
        output = torch.matmul(scores,v)
        return output
    
    def forward(self, q, k, v, mask = None):
        batchsize = q.size(0)
        # 转置操作 .transpose(1, 2) 是为了在多头注意力计算中正确对齐每个头的查询、键和值
        # 矩阵计算在sequence_length, d_k这两个维度上进行
        # -1: 序列长度
        k = self.k_linear(k).view(batchsize, -1, self.h, self.d_k).transpose(1, 2)
        q = self.q_linear(q).view(batchsize, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batchsize, -1, self.h, self.d_k).transpose(1, 2)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1, 2).contiguous().view(batchsize, -1, self.d_model)
        output = self.out(concat)
        return output