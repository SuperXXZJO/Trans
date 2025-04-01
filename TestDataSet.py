import torch
import torch.nn as nn
import random
import string

# 1. 构建 vocab 字典
all_chars = list(string.ascii_lowercase) + ['<pad>']  # 加上 padding token
char2idx = {ch: i for i, ch in enumerate(all_chars)}
idx2char = {i: ch for ch, i in char2idx.items()}
vocab_size = len(char2idx)

# 2. 生成 toy 数据（逆序）
def generate_reverse_data(num_samples=1000, seq_len=5):
    data = []
    for _ in range(num_samples):
        s = ''.join(random.choices(string.ascii_lowercase, k=seq_len))
        reversed_s = s[::-1]
        data.append((s, reversed_s))
    return data

# 3. 编码字符串为 token ID 序列
def encode_sequence(s, max_len):
    token_ids = [char2idx[c] for c in s]
    if len(token_ids) < max_len:
        token_ids += [char2idx['<pad>']] * (max_len - len(token_ids))
    return token_ids

# 4. 构建 embedding 层
embedding_dim = 16
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# 5. 把数据转换成 embedding
def prepare_embedding_data(data, max_len):
    input_ids = [encode_sequence(inp, max_len) for inp, _ in data]
    target_ids = [encode_sequence(tgt, max_len) for _, tgt in data]

    input_ids = torch.LongTensor(input_ids)     # shape: [batch, seq]
    target_ids = torch.LongTensor(target_ids)

    input_embed = embedding_layer(input_ids)    # shape: [batch, seq, embedding_dim]
    target_embed = embedding_layer(target_ids)

    return input_embed, target_embed, input_ids, target_ids  # 可同时返回 ID 方便调试

data = generate_reverse_data(num_samples=10, seq_len=6)
input_embed, target_embed, input_ids, target_ids = prepare_embedding_data(data, max_len=6)


print("原始数据对：", data[0])
print("输入 token id：", input_ids[0])
print("嵌入向量：", input_embed[0].shape)  # 应该是 [seq_len, embedding_dim]