import torch
import numpy as np
import torch.nn as nn


class ScaledDotproductAttention(nn.Module):
    def __init__(self, dropout=0.):
        super(ScaledDotproductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, mask=None):
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention *= scale
        if mask:
            attention = attention.masked_fill_(mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return attention, context


if __name__ == '__main__':
    text = torch.randn(2, 2, 3)
    att, con = ScaledDotproductAttention()(text, text, text)
    print(att)
    print(con)
