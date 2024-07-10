import torch.nn as nn
from multi_head_attention import MultiHeadAttention as Msa
from ffn import FeedForwardBlock as Ffb
from residual_connection import ResidualConnection as Rc


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: Msa, feed_forward_block: Ffb, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([Rc(dropout), Rc(dropout)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[0](x, self.feed_forward_block)
        return x