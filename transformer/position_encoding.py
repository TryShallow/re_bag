import torch
import numpy as np
import torch.nn as nn


class PositionEncoding(nn.Module):
    def __init__(self, max_seq_len=64, d_model=512):
        super(PositionEncoding, self).__init__()
        pe = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)
        ], dtype=np.float32)
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        pe = np.pad(pe, ((1, 0), (0, 0)), mode='constant')
        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(torch.tensor(pe), False)

    def forward(self, x):
        max_len = torch.max(x).item()
        tensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        x_pos = tensor(
            [list(range(1, len.item() + 1)) + [0] * (max_len - len.item()) for len in x.data]
        )
        return self.position_encoding(x_pos)


if __name__ == '__main__':
    pe = PositionEncoding()
    x = torch.tensor([1, 6, 2]).view(-1, 1)
    print(x.size())
    for i in x.data:
        print(i.item())
    print(pe(x).size())
    # print([0] * torch.max(x))
    # print([list(range(1, len.item() + 1)) + [0] * (4 - len.item()) for len in x.data])