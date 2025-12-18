import torch.nn as nn
import torch.nn.functional as F

class Attn_Net_Gated(nn.Module):
    def __init__(self, hidden_dim, head_dim, dropout, n_classes=4):
        super().__init__()

        att_a = [nn.Linear(hidden_dim, head_dim), nn.Tanh()]

        att_b = [nn.Linear(hidden_dim, head_dim), nn.Sigmoid()]

        if dropout:
            att_a.append(nn.Dropout(dropout))
            att_b.append(nn.Dropout(dropout))

        self.attention_a = nn.Sequential(*att_a)
        self.attention_b = nn.Sequential(*att_b)
        self.attention_c = nn.Linear(head_dim, n_classes)

    def forward(self, x):
        # x : B x N x hidden_dim
        a = self.attention_a(x) # a : B x N x head_dim
        b = self.attention_b(x) # b : B x N x head_dim
        A = a.mul(b) # A : B x N x head_dim
        A = self.attention_c(A)  # A : B X N x n_classes
        return A
