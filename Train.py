import torch
import torch.nn as nn
import torch.optim as optim
from Transformer import Transformer

import TestDataSet

# Hyperparameters
d_model = 16
N_layer = 2
heads = 2
dropout = 0.1
lr = 0.001
epoch = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = TestDataSet.generate_reverse_data(num_samples=10, seq_len=6)
input_embed, target_embed, input_ids, target_ids = TestDataSet.prepare_embedding_data(data, max_len=6)


model = Transformer(input_embed,target_embed,d_model,N_layer,heads,dropout)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr)


for epoch in range(10):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for i in range(1000):
        inputs, labels = input_embed[i], target_embed[i]

        outputs = model(inputs,labels,False,True)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}, Accuracy = {acc:.2f}%")