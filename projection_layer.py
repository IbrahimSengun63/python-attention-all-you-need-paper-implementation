import torch
import torch.nn as nn


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.projection_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch,seq,d_model)--> (batch,seq,vocab_size)
        return torch.log_softmax(self.projection_layer[x], dim=-1)
