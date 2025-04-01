import torch
import torch.nn as nn
import torch.nn.functional as Func
import math

from SubLayer.EncoderLayer import EncoderLayer

class Encoder(nn.Module):

    def __init__(self,src,d_model, N_layer, heads, dropout):
        super().__init__()
        self.N = N_layer
        # 保留了embbeding
        # self.embed = nn.Embedding(vocab_size, d_model)
        self.src = src
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, dropout) for _ in range(N_layer)])
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self,mask):
        # x = self.embed(src)
        x = self.src
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)