import torch.nn as nn
from layer_normalization import LayerNormalization


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        normalized_x = self.norm(x)
        sublayer_output = sublayer(normalized_x)
        dropped_out = self.dropout(sublayer_output)
        return x + dropped_out
