import torch
import torch.nn as nn
import torch.nn.functional as Func
import math

from SubLayer.MultiHeadAttention import MultiheadAttention
from SubLayer.FeedForward import FeedForward

class Decoderlayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm_3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.attention_1 = MultiheadAttention(heads, d_model, dropout=dropout)
        self.attention_2 = MultiheadAttention(heads, d_model, dropout=dropout)
        self.feedforward = FeedForward(d_model, dropout=dropout)

    def forforward(self,x,e_outputs,src_mask,trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attention_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attention_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.feedforward(x2))
        return x