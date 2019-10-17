import torch
import torch.nn as nn

from scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, dropout=0.):
        super(MultiHeadAttention, self).__init__()
        self.dim_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linears = nn.ModuleList([nn.Linear(model_dim, self.dim_head * num_heads)] * 3)
        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None):
        residual = query
        batch_size = key.size(0)
        query = self.linears[0](query)
        key = self.linears[1](key)
        value = self.linears[2](value)

        query = query.view(batch_size * self.num_heads, -1, self.dim_head)
        key = key.view(batch_size * self.num_heads, -1, self.dim_head)
        value = value.view(batch_size * self.num_heads, -1, self.dim_head)

        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1)
        scale = (key.size(-1) // self.num_heads) ** -0.5
        att, con = self.dot_product_attention(query, key, value, scale, mask)
        con = con.view(batch_size, -1, self.dim_head * self.num_heads)
        output = self.linear_final(con)
        output = self.dropout(output)
        output = self.layer_norm(residual + output)

        return output, att


def padding_mask(seq_query, seq_key):
    len_query = seq_query.size(1)
    pad_mask = seq_key.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_query, -1)
    return pad_mask


def sequence_mask(seq):
    bs, sl = seq.size()
    mask = torch.triu(torch.ones((sl, sl), dtype=torch.uint8), diagonal=1)
    mask = mask.unsqueeze(0).expand(bs, -1, -1)
    return mask


if __name__ == '__main__':
    q = torch.randn(2, 4, 512)
    o, att = MultiHeadAttention()(q, q, q)
    print(o.size())
    # print(torch.triu(torch.ones((4, 4)), 1))