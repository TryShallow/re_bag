import torch
import torch.nn as nn

from multi_head_attention import MultiHeadAttention, padding_mask, sequence_mask
from position_wise_feed_forward import PositionWiseFeedForward
from position_encoding import PositionEncoding


class DecoderLayer(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=204, dropout=0.0):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.joint_attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, dec_inputs, enc_outputs, self_attn_mask=None, context_attn_mask=None):
        dec_outputs, self_att = self.self_attention(dec_inputs, dec_inputs, dec_inputs, self_attn_mask)
        dec_outputs, con_att = self.joint_attention(dec_outputs, enc_outputs, enc_outputs, context_attn_mask)
        dec_outputs = self.feed_forward(dec_outputs)
        return dec_outputs, self_att, con_att


class Decoder(nn.Module):
    def __init__(self, vocab_size, max_seq_len, num_layers=6, model_dim=512, num_heads=8,
                 ffn_dim=2048, dropout=0.0):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
        self.pos_embedding = PositionEncoding(max_seq_len, model_dim)

    def forward(self, x, x_len, enc_output, context_att_mask=None):
        output = self.seq_embedding(x)
        output += self.pos_embedding(x_len)

        self_att_padding_mask = padding_mask(x, x)
        seq_mask = sequence_mask(x)
        self_att_mask = torch.gt(self_att_padding_mask + seq_mask, 0)
        self_atts = []
        con_atts = []
        for decoder in self.decoder_layers:
            output, self_att, con_att = decoder(output, enc_output, self_att_mask, context_att_mask)
            self_atts.append(self_att)
            con_atts.append(con_att)
        return output, self_atts, con_atts


if __name__ == '__main__':
    d = Decoder(200, 20)
    from encoder import Encoder
    encoder = Encoder(200, 20)
    x = (torch.randn(2, 6) * 10 // 1 % 10).type(torch.int64)
    x[x < 0] = 0
    x_len = torch.tensor([1, 1]).view(-1, 1)
    enc_output, _ = encoder(x, x_len)
    dec_output, _, _ = d(x, x_len, enc_output)
    print(dec_output.size())