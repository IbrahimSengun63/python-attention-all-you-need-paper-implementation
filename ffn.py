import torch
import torch.nn as nn


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # W2 and B2

    def forward(self, x):
        # (batch,seq_len,d_model) --> (batch,seq_len,d_ff) --> (batch,seq_len,d_model)
        x = self.linear_1(x)  # Apply first linear transformation
        x = torch.relu(x)  # Apply ReLU activation
        x = self.dropout(x)  # Apply dropout
        x = self.linear_2(x)  # Apply second linear transformation
        return x
