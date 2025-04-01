import torch
import torch.nn as nn
import torch.nn.functional as Func
import math

from SubLayer.MultiHeadAttention import MultiheadAttention
from SubLayer.FeedForward import FeedForward
# encoder unit
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.attention = MultiheadAttention(heads, d_model, dropout=dropout)
        self.feedforward = FeedForward(d_model, dropout=dropout)
        # todo not sure if needed
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    # todo 有争议 不确定什么时候dropout 和 norm
    # now is pre-norm
    def forward(self,x,mask):
        x2 = self.layernorm_1(x)
        x = x + self.dropout_1(self.attention(x2, x2, x2, mask))
        x2 = self.layernorm_2(x)
        x = x + self.dropout_2(self.feedforward(x2))
        return x