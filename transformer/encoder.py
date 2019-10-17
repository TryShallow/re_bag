import torch
import torch.nn as nn

from position_encoding import PositionEncoding
from multi_head_attention import MultiHeadAttention, padding_mask
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


class Encoder(nn.Module):
    def __init__(self, vocab_size, max_seq_len, num_layers=6, model_dim=512, num_heads=8,
                 ffn_dim=2048, dropout=0.0):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
        self.pos_embedding = PositionEncoding(max_seq_len, model_dim)

    def forward(self, x, x_len):
        output = self.seq_embedding(x)
        output += self.pos_embedding(x_len)
        self_att_mask = padding_mask(x, x)
        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_att_mask)
            attentions.append(attention)
        return output, attentions


if __name__ == '__main__':
    el = EncoderLayer()
    x = torch.randn(2, 3, 512)
    print(el(x)[0].size())

    encoder = Encoder(200, 20)
    x = (torch.randn(2, 3) * 10 // 1 % 10).type(torch.int64)
    x[x < 0] = 0
    print(x)
    x_len = torch.tensor([1, 1]).view(-1, 1)
    print(encoder(x, x_len)[0].size())