import math

import torch.nn as nn
import torch


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create matrix (seq_len,d_model)
        position_matrix = torch.zeros(seq_len, d_model)

        # Create vector shape (seq_len,1) --> numerator
        numerator = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # paper position encoding formula denominator part
        denominator = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # multiply with sin at even position
        position_matrix[:, 0::2] = torch.sin(numerator * denominator)

        # multiply with cos at odd position
        position_matrix[:, 1::2] = torch.cos(numerator * denominator)

        # add batch dimension to the position matrix new shape --> (1,seq_len,d_model)
        position_matrix = position_matrix.unsqueeze(0)

        # save the position matrix alongside with model as buffer meaning not as a learned parameter
        self.register_buffer('position_matrix', position_matrix)

    def forward(self, x):
        # adding position matrix to the every input embedding
        x = x + self.position_matrix[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)
