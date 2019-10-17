import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder
from multi_head_attention import padding_mask, sequence_mask


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, src_max_len, tgt_vocab_size, tgt_max_len, num_layers=6,
                 model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.2):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, src_max_len, num_layers, model_dim, num_heads,
                               ffn_dim, dropout)
        self.decoder = Decoder(tgt_vocab_size, tgt_max_len, num_layers, model_dim, num_heads,
                               ffn_dim, dropout)
        self.linear = nn.Linear(model_dim, tgt_vocab_size, False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, src_seq, src_len, tgt_seq, tgt_len):
        con_att_mask = padding_mask(src_seq, tgt_seq)
        output, enc_self_att = self.encoder(src_seq, src_len)
        output, dec_self_att, con_att = self.decoder(tgt_seq, tgt_len, output, con_att_mask)
        output = self.linear(output)
        output = self.softmax(output)
        return output, enc_self_att, dec_self_att, con_att


if __name__ == '__main__':
    x = (torch.randn(2, 3) * 10 // 1 % 10).type(torch.int64)
    x[x < 0] = 0
    x_len = torch.tensor([1, 1]).view(-1, 1)
    tf = Transformer(200, 20, 200, 20)
    out, _, _, _ = tf(x, x_len, x, x_len)
    print(out.size())
