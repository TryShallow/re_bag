import torch
import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(model_dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.w2(self.relu(self.w1(x)))
        output = self.dropout(output)
        output = self.layer_norm(x + output)
        return output


if __name__ == '__main__':
    p = PositionWiseFeedForward()
    x = torch.randn(2, 4, 512)
    print(p(x).size())
