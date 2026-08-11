from __future__ import annotations

import torch
from torch import Tensor, nn


class FourierFeatures(nn.Module):
    def __init__(self, input_dim: int, num_bands: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_bands = int(num_bands)
        frequencies = 2.0 ** torch.arange(self.num_bands, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)

    def output_dim(self) -> int:
        return self.input_dim * (1 + 2 * self.num_bands)

    def forward(self, values: Tensor) -> Tensor:
        if values.shape[-1] != self.input_dim:
            raise ValueError(f"Expected last dim {self.input_dim}, got {values.shape[-1]}")
        expanded = values.unsqueeze(-1) * self.frequencies.view(*((1,) * values.ndim), -1)
        sin = torch.sin(2.0 * torch.pi * expanded)
        cos = torch.cos(2.0 * torch.pi * expanded)
        return torch.cat([values, sin.flatten(-2), cos.flatten(-2)], dim=-1)


class ContinuousMultiDimRoPE(nn.Module):
    def __init__(
        self,
        head_dim: int,
        coordinate_dims: list[int],
        coordinate_scale: float,
        rope_base: float,
    ) -> None:
        super().__init__()
        if sum(int(value) for value in coordinate_dims) != int(head_dim):
            raise ValueError("coordinate_dims must sum to head_dim")
        if any(int(value) <= 0 or int(value) % 2 != 0 for value in coordinate_dims):
            raise ValueError("Every coordinate RoPE slice must be a positive even dimension")
        self.head_dim = int(head_dim)
        self.coordinate_dims = [int(value) for value in coordinate_dims]
        self.coordinate_scale = float(coordinate_scale)
        self.rope_base = float(rope_base)
        inv_freq = []
        for dim in self.coordinate_dims:
            half = dim // 2
            inv = 1.0 / (self.rope_base ** (torch.arange(half, dtype=torch.float32) / max(1, half)))
            inv_freq.append(inv)
        self.inv_freq = nn.ParameterList([nn.Parameter(item, requires_grad=False) for item in inv_freq])

    @staticmethod
    def _rotate_pairs(values: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        batch, heads, tokens, dim = values.shape
        pairs = values.reshape(batch, heads, tokens, dim // 2, 2)
        even = pairs[..., 0]
        odd = pairs[..., 1]
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        return torch.stack([rotated_even, rotated_odd], dim=-1).reshape(batch, heads, tokens, dim)

    def forward(self, query: Tensor, key: Tensor, coordinates: Tensor) -> tuple[Tensor, Tensor]:
        if coordinates.shape[-1] != len(self.coordinate_dims):
            raise ValueError(
                f"Expected {len(self.coordinate_dims)} coordinate channels, got {coordinates.shape[-1]}"
            )
        query_chunks = []
        key_chunks = []
        start = 0
        for coord_index, dim in enumerate(self.coordinate_dims):
            end = start + dim
            coord = coordinates[..., coord_index].unsqueeze(1).unsqueeze(-1) * self.coordinate_scale
            angles = coord * self.inv_freq[coord_index].view(1, 1, 1, -1)
            cos = torch.cos(angles)
            sin = torch.sin(angles)
            query_chunks.append(self._rotate_pairs(query[..., start:end], cos, sin))
            key_chunks.append(self._rotate_pairs(key[..., start:end], cos, sin))
            start = end
        return torch.cat(query_chunks, dim=-1), torch.cat(key_chunks, dim=-1)


class FeedForward(nn.Module):
    def __init__(self, width: int, hidden_dim: int, dropout: float, activation: str = "gelu") -> None:
        super().__init__()
        activation_layer = nn.GELU() if str(activation).lower() == "gelu" else nn.SiLU()
        self.network = nn.Sequential(
            nn.Linear(width, hidden_dim),
            activation_layer,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, width),
            nn.Dropout(dropout),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float, rope: ContinuousMultiDimRoPE | None = None) -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError(f"width {width} must be divisible by heads {heads}")
        self.width = int(width)
        self.heads = int(heads)
        self.head_dim = self.width // self.heads
        self.dropout = nn.Dropout(dropout)
        self.qkv = nn.Linear(self.width, self.width * 3)
        self.output = nn.Linear(self.width, self.width)
        self.rope = rope

    def forward(self, values: Tensor, valid_mask: Tensor | None = None, coordinates: Tensor | None = None) -> Tensor:
        batch, length, _ = values.shape
        qkv = self.qkv(values).reshape(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        if self.rope is not None:
            if coordinates is None:
                raise ValueError("RoPE attention requires coordinates")
            query, key = self.rope(query, key, coordinates)
        logits = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if valid_mask is not None:
            key_invalid = ~valid_mask[:, None, None, :]
            logits = logits.masked_fill(key_invalid, torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=-1)
        attention = self.dropout(attention)
        attended = torch.matmul(attention, value).transpose(1, 2).reshape(batch, length, self.width)
        output = self.output(attended)
        if valid_mask is not None:
            output = output * valid_mask.unsqueeze(-1).to(output.dtype)
        return output


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError(f"width {width} must be divisible by heads {heads}")
        self.width = int(width)
        self.heads = int(heads)
        self.head_dim = self.width // self.heads
        self.dropout = nn.Dropout(dropout)
        self.query = nn.Linear(self.width, self.width)
        self.key = nn.Linear(self.width, self.width)
        self.value = nn.Linear(self.width, self.width)
        self.output = nn.Linear(self.width, self.width)

    def forward(
        self,
        query_tokens: Tensor,
        context_tokens: Tensor,
        query_mask: Tensor | None = None,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        batch, q_len, _ = query_tokens.shape
        k_len = context_tokens.shape[1]
        query = self.query(query_tokens).reshape(batch, q_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        key = self.key(context_tokens).reshape(batch, k_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        value = self.value(context_tokens).reshape(batch, k_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        logits = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if context_mask is not None:
            context_invalid = ~context_mask[:, None, None, :]
            logits = logits.masked_fill(context_invalid, torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=-1)
        attention = self.dropout(attention)
        attended = torch.matmul(attention, value).transpose(1, 2).reshape(batch, q_len, self.width)
        output = self.output(attended)
        if query_mask is not None:
            output = output * query_mask.unsqueeze(-1).to(output.dtype)
        return output


class SelfAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int, ffn_dim: int, dropout: float, rope: ContinuousMultiDimRoPE | None = None, activation: str = "gelu") -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(width)
        self.attn = MultiHeadSelfAttention(width, heads, dropout, rope=rope)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = FeedForward(width, ffn_dim, dropout, activation=activation)

    def forward(self, values: Tensor, valid_mask: Tensor | None = None, coordinates: Tensor | None = None) -> Tensor:
        values = values + self.attn(self.attn_norm(values), valid_mask=valid_mask, coordinates=coordinates)
        values = values + self.ffn(self.ffn_norm(values))
        if valid_mask is not None:
            values = values * valid_mask.unsqueeze(-1).to(values.dtype)
        return values


class CrossAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int, ffn_dim: int, dropout: float, activation: str = "gelu") -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width)
        self.cross_attn = MultiHeadCrossAttention(width, heads, dropout)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = FeedForward(width, ffn_dim, dropout, activation=activation)

    def forward(
        self,
        query_tokens: Tensor,
        context_tokens: Tensor,
        query_mask: Tensor | None = None,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        query_tokens = query_tokens + self.cross_attn(
            self.query_norm(query_tokens),
            self.context_norm(context_tokens),
            query_mask=query_mask,
            context_mask=context_mask,
        )
        query_tokens = query_tokens + self.ffn(self.ffn_norm(query_tokens))
        if query_mask is not None:
            query_tokens = query_tokens * query_mask.unsqueeze(-1).to(query_tokens.dtype)
        return query_tokens
