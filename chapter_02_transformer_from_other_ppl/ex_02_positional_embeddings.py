import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



vocab_size = 10
d_model = 64   # embedding sizes
embed = nn.Embedding(vocab_size, d_model)


max_seq_len = 800

pe = np.zeros(shape=(max_seq_len, d_model))

for pos in range(max_seq_len):
    for i in range(0, d_model, 2):
        arg = pos / (10000 ** ((2 * i) / d_model))
        pe[pos, i] = np.sin(arg)
        arg2 = pos / (10000 ** ((2 * (i + 1)) / d_model))
        pe[pos, i + 1] = np.cos(arg2)

heatmap = sns.heatmap(pe)
plt.show()