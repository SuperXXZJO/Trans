import torch
import torch.nn as nn

d_model = 5
k_linear = nn.Linear(d_model, d_model)


k = torch.randn(10, 2)  # 假设有10个样本，每个样本有d_model个特征
print(k)

output = k_linear(k)

print(output)