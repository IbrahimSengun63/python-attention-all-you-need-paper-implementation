import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (batch,h,d_k,seq) -- > key after transpose
        attention_scores = (query @ key.transope(-2, -1)) / math.sqrt(d_k)

        # Masking future words or padding words
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)  # (batch,h,seq,seq)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # following three lines perform matrix multiplication with q(batch,seq,d_model) @ Wq(batch,d_model,d_model)
        query = self.w_q(q)  # (Batch,seq,d_model) --> (Batch,seq,d_model)
        key = self.w_q(k)  # (Batch,seq,d_model) --> (Batch,seq,d_model)
        value = self.w_q(v)  # (Batch,seq,d_model) --> (Batch,seq,d_model)

        # following three lines split query key and value according to the h and d_k
        #  (Batch,seq,d_model) --> (Batch,seq,d_k) --> (batch,h,seq,d_k)
        query = query.view(query.shape[0], query[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value[1], self.h, self.d_k).transpose(1, 2)

        # perform attention and return both attention values and output attention score
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (batch,h,seq,d_k) --> (batch,seq,h,d_k) --> (batch,seq,d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        # (Batch,seq,d_model) --> (Batch,seq,d_model)
        return self.w_o(x)
