import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, channels, n_heads, dropout=0.0, window_size=None):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.window_size = window_size

        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)

        self.dropout = dropout

    def forward(self, x, mask=None):
        batch_size, seq_len, channels = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0) - torch.arange(seq_len, device=x.device).unsqueeze(1)
        positions = positions.clamp(-self.max_relative_position, self.max_relative_position) + self.max_relative_position

        relative_key_emb = self.relative_key[positions]
        relative_scores = torch.einsum('bhqd,qkd->bhqk', q, relative_key_emb) / math.sqrt(self.head_dim)

        scores = scores + relative_scores

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)

        if self.dropout > 0:
            attn_weights = F.dropout(attn_weights, self.dropout, self.training)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)

        output = self.out_proj(attn_output)

        return output


class RelativePositionMultiHeadAttention(nn.Module):
    def __init__(self, channels, n_heads, dropout=0.0, max_relative_position=100):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.max_relative_position = max_relative_position

        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)

        self.relative_key = nn.Parameter(torch.randn(2 * max_relative_position + 1, self.head_dim))
        self.relative_value = nn.Parameter(torch.randn(2 * max_relative_position + 1, self.head_dim))

        self.dropout = dropout

    def forward(self, x, mask=None):
        batch_size, seq_len, channels = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0) - torch.arange(seq_len, device=x.device).unsqueeze(1)
        positions = positions.clamp(-self.max_relative_position, self.max_relative_position) + self.max_relative_position

        relative_key_emb = self.relative_key[positions]
        relative_scores = torch.einsum('bhqd,qkd->bhqk', q, relative_key_emb) / math.sqrt(self.head_dim)

        scores = scores + relative_scores

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)

        attn_weights = F.softmax(scores, dim=-1)

        if self.dropout > 0:
            attn_weights = F.dropout(attn_weights, self.dropout, self.training)

        attn_output = torch.matmul(attn_weights, v)

        relative_value_emb = self.relative_value[positions]
        relative_attn = torch.einsum('bhqk,qkd->bhqd', attn_weights, relative_value_emb)
        attn_output = attn_output + relative_attn

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)

        output = self.out_proj(attn_output)

        return output


class FeedForward(nn.Module):
    def __init__(self, channels, filter_channels, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(channels, filter_channels)
        self.fc2 = nn.Linear(filter_channels, channels)
        self.dropout = dropout

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, self.training)
        x = self.fc2(x)
        return x
