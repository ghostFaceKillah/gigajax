import math

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



vocab_size = 10
d_model = 64   # embedding sizes
embed = nn.Embedding(vocab_size, d_model)



class PositionalEncoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 max_seq_len: int = 80):
        super().__init__()
        self.d_model = d_model

        # create constant `pe` matrix  with values dep on pos and i
        # pos goes through max len of sequence, i goes through model dims

        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False)
        # very strange
        # very, very strange
        return x

print("Whateva")