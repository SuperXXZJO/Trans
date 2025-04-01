import torch.nn as nn
from SubLayer.DecoderLayer import Decoderlayer

class Decoder(nn.Module):

    def __init__(self,trg,d_model, N_layer, heads, dropout):
        super().__init__()
        self.N = N_layer
        # self.embed = nn.Embedding(vocab_size, d_model)
        self.trg = trg
        self.layers = nn.ModuleList([Decoderlayer(d_model, heads, dropout) for _ in range(N_layer)])
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, e_outputs, src_mask, trg_mask):
        # x = self.embed(trg)
        x = self.trg
        for layer in self.layers:
            x = layer(x, e_outputs, src_mask, trg_mask)
        return self.norm(x)