import torch
from torch import nn, einsum
from torch.nn import Conv1d
from torch import Tensor
from einops.layers.torch import Rearrange
from einops import rearrange
from torch.nn import functional as F
import numpy as np
import math
from typing import Union

# TODO Check activate function for the XAI methods


class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x


def ConvBlock(
    in_channels,
    out_channels,
    kernel_size,
    padding: Union[str, int],
    stride=1,
    dilation=1,
    bias=True,
    batch_norm=True,
):
    return nn.Sequential(
        nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            bias=bias,
        ),
        # nn.BatchNorm1d(out_channels) if batch_norm else nn.Identity(),
        nn.ReLU(),
        nn.BatchNorm1d(out_channels) if batch_norm else nn.Identity(),
    )


# from https://github.com/boxiangliu/enformer-pytorch/blob/main/model/enformer.py
class Residual(nn.Module):
    """residual block"""

    def __init__(self, module):
        super().__init__()
        self._module = module

    def forward(self, x, *args, **kwargs):
        return x + self._module(x, *args, **kwargs)


class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size=2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange("b d (n p) -> b d n p", p=2)
        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = F.pad(x, (0, remainder), value=0)
            mask = torch.zeros((b, 1, n), dtype=torch.bool, device=x.device)
            mask = F.pad(mask, (0, remainder), value=True)

        x = self.pool_fn(x)
        logits = self.to_attn_logits(x)

        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)

        attn = logits.softmax(dim=-1)

        return (x * attn).sum(dim=-1)

class SoftmaxPool(nn.Module):
    def __init__(self, pool_size=2):
        super().__init__()
        assert isinstance(pool_size, int) and pool_size > 0, "pool_size should be a positive integer"
        self.pool_size = pool_size
        self.pool_fn = Rearrange("b c (n p) -> b c n p", p=pool_size)

    def forward(self, x):
        assert isinstance(x, torch.Tensor) and x.ndim == 3, "Input tensor should be a 3D tensor of shape (batch_size, input_channels, sequence_length)"
        batch_size, input_channels, sequence_length = x.shape

        # Apply reflection padding to the input tensor
        remainder = sequence_length % self.pool_size
        padding = 0
        if remainder > 0:
            padding = self.pool_size - remainder
            x = F.pad(x, (0, padding), mode='reflect')

        # Create a mask tensor to identify the padded elements
        mask = torch.zeros((batch_size, 1, sequence_length + padding), dtype=torch.bool, device=x.device)
        if padding > 0:
            mask[:, :, -padding:] = True

        # Rearrange the dimensions of the input tensor
        x = self.pool_fn(x)

        # Compute the attention weights using softmax
        logits = x
        mask_value = -torch.finfo(logits.dtype).max
        if padding > 0:
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)
        attn = logits.softmax(dim=-1)

        # Weight the pooled tensor elements using the attention weights
        output = (x * attn).sum(dim=-1)

        return output


# For Transformer -------------------------------------------------------------------------------------

# Positional encoding version #2 (from Attention is all you need) ------------------
def get_angles(pos, i, encoding_dim):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(encoding_dim))
    return pos * angle_rates


def positional_encoding_v2(position, encoding_dim):
    """
    Args:
        position: the number of positions in the input sequence
    """
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(encoding_dim)[np.newaxis, :],
        encoding_dim,
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads
    return pos_encoding


# Posional encoding version #1 (from enformer-pytorch) -----------------------
def exists(val):
    return val is not None


def get_positional_features_exponential(
    positions, features, seq_len, min_half_life=3.0
):
    max_range = math.log(seq_len) / math.log(2.0)
    half_life = 2 ** torch.linspace(
        min_half_life, max_range, features, device=positions.device
    )
    half_life = half_life[None, ...]
    positions = positions.abs()[..., None]
    return torch.exp(-math.log(2.0) / half_life * positions)


def get_positional_features_central_mask(positions, features, seq_len):
    center_widths = 2 ** torch.arange(1, features + 1, device=positions.device).float()
    center_widths = center_widths - 1
    return (center_widths[None, ...] > positions.abs()[..., None]).float()


def gamma_pdf(x, concentration, rate):
    log_unnormalized_prob = torch.xlogy(concentration - 1.0, x) - rate * x
    log_normalization = torch.lgamma(concentration) - concentration * torch.log(rate)
    return torch.exp(log_unnormalized_prob - log_normalization)


def get_positional_features_gamma(
    positions, features, seq_len, stddev=None, start_mean=None, eps=1e-8
):
    if not exists(stddev):
        stddev = seq_len / (2 * features)

    if not exists(start_mean):
        start_mean = seq_len / features

    mean = torch.linspace(start_mean, seq_len, features, device=positions.device)
    mean = mean[None, ...]
    concentration = (mean / stddev) ** 2
    rate = mean / stddev ** 2
    probabilities = gamma_pdf(positions.float().abs()[..., None], concentration, rate)
    probabilities = probabilities + eps
    outputs = probabilities / torch.amax(probabilities, dim=-1, keepdim=True)
    return outputs


def positional_encoding_v1(seq_len, feature_size, device):
    distances = torch.arange(-seq_len + 1, seq_len, device=device)

    feature_functions = [
        get_positional_features_exponential,
        get_positional_features_central_mask,
        get_positional_features_gamma,
    ]

    num_components = len(feature_functions) * 2

    if (feature_size % num_components) != 0:
        raise ValueError(
            f"feature size is not divisible by number of components ({num_components})"
        )

    num_basis_per_class = feature_size // num_components

    embeddings = []
    for fn in feature_functions:
        embeddings.append(fn(distances, num_basis_per_class, seq_len))

    embeddings = torch.cat(embeddings, dim=-1)
    embeddings = torch.cat(
        (embeddings, torch.sign(distances)[..., None] * embeddings), dim=-1
    )
    return embeddings


# Attention Layer version #1 (Use positional encoding version #1) --------------------------------------------------
def relative_shift(x):
    to_pad = torch.zeros_like(x[..., :1])
    x = torch.cat((to_pad, x), dim=-1)
    _, h, t1, t2 = x.shape
    x = x.reshape(-1, h, t2, t1)
    x = x[:, :, 1:, :]
    x = x.reshape(-1, h, t1, t2 - 1)
    return x[..., : ((t2 + 1) // 2)]


class Attention(nn.Module):
    def __init__(
        self,
        dim,  # the input has shape (batch_size, len, dim) = (b, n, dim)
        *,
        heads=8,
        dim_key=64,
        dim_value=64,
        dropout=0.0,
        pos_dropout=0.0,
        num_rel_pos_features=66,
    ):
        super().__init__()
        self.scale = dim_key ** -0.5
        self.heads = heads

        # Q, K, V

        self.to_q = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_value * heads, bias=False)

        self.to_out = nn.Linear(dim_value * heads, dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        # relative positional encoding

        self.num_rel_pos_features = num_rel_pos_features  ###########

        self.to_rel_k = nn.Linear(
            self.num_rel_pos_features, dim_key * heads, bias=False
        )

        self.rel_content_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))
        self.rel_pos_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))

        # dropouts

        self.pos_dropout = nn.Dropout(pos_dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        n, h, device = x.shape[-2], self.heads, x.device

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        q = q * self.scale

        content_logits = einsum(
            "b h i d, b h j d -> b h i j", q + self.rel_content_bias, k
        )

        positions = positional_encoding_v1(n, self.num_rel_pos_features, device)
        positions = self.pos_dropout(positions)
        rel_k = self.to_rel_k(positions)

        rel_k = rearrange(rel_k, "n (h d) -> h n d", h=h)
        rel_logits = einsum("b h i d, h j d -> b h i j", q + self.rel_pos_bias, rel_k)
        rel_logits = relative_shift(rel_logits)

        logits = content_logits + rel_logits  # shape (b, h, n, n)
        attn = logits.softmax(dim=-1)  # softmax over the last dimension
        attn = self.attn_dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)  # (b, n, dim)


# Attention Layer version #2 ------------------------------------------------------------
# Positional encoding is added to the input only once before attention layers
# This attention layer is used in Transformer_2 and Transformer_3
class Attention_2(nn.Module):
    def __init__(
        self,
        dim,  # the input has shape (batch_size, len, dim) = (b, n, dim)
        *,
        heads=8,
        dim_key=64,
        dim_value=64,
        dropout=0.0,
        pos_dropout=0.0,
    ):
        super().__init__()
        self.scale = dim_key ** -0.5
        self.heads = heads
        self.dim = dim

        # Q, K, V

        self.to_q = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_value * heads, bias=False)

        self.to_out = nn.Linear(dim_value * heads, dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        self.rel_content_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))

        # dropouts

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        n, h, device = x.shape[-2], self.heads, x.device

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        q = q * self.scale

        logits = einsum("b h i d, b h j d -> b h i j", q + self.rel_content_bias, k)
        attn = logits.softmax(dim=-1)  # softmax over the last dimension
        attn = self.attn_dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)  # (b, n, dim)
