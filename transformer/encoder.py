import torch
import torch.nn as nn

from multi_head_attention import MultiHeadAttention
from position_wise_feed_forward import PositionWiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, x, mask=None):
        con, att = self.attention(x, x, x, mask)
        output = self.feed_forward(con)
        return output, att


if __name__ == '__main__':
    el = EncoderLayer()
    x = torch.randn(2, 3, 512)
    print(el(x)[0].size())