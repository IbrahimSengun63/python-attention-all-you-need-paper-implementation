import math

import torch.nn as nn


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # it will create embeddings vector based on d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    # it will fill the embedding vector with the input * its weights sqrt
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
