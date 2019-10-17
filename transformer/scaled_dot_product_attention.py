import torch
import numpy as np
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, mask=None):
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention *= scale
        if mask is not None:
            attention = attention.masked_fill_(mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return attention, context


if __name__ == '__main__':
    text = torch.randn(2, 3, 4)
    att, con = ScaledDotProductAttention()(text, text, text)
    print(att)
    print(con)
