import torch
import torch.nn.functional as F
import math
import numpy as np

class ChebyshevKANLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyshevKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = torch.nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        torch.nn.init.xavier_normal_(self.cheby_coeffs)
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def chebyshev_polynomials(self, x):
        T = [torch.ones_like(x), 2*x]
        for n in range(2, self.degree + 1):
            T.append(2 * x * T[n - 1] - T[n - 2])
        return torch.stack(T, dim=-1)

    def forward(self, x):
        x = x.view(-1, self.inputdim)
        x = torch.tanh(x)
        T = self.chebyshev_polynomials(x)
        y = torch.einsum("bij,ioj->bo", T, self.cheby_coeffs)
        y = y.view(-1, self.outdim)
        return y

class KAN(torch.nn.Module):
    def __init__(self, layers_hidden, degree=3):
        super(KAN, self).__init__()

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                ChebyshevKANLayer(
                    in_features,
                    out_features,
                    degree,
                )
            )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

class MultiheadKANAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, rotation_matrix, degree=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.position_emb = rotation_matrix

        self.qkv_linear = ChebyshevKANLayer(hidden_size, hidden_size * 3, degree)
        self.out = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size, seq_length, hidden_size = x.size()

        qkv = self.qkv_linear(x)

        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.transpose(1, 2)

        queries, keys, values = qkv.chunk(3, dim=-1)
        queries = apply_rotary_pos_emb(queries, self.position_emb)
        keys = apply_rotary_pos_emb(keys, self.position_emb)

        scores = torch.matmul(queries, keys.transpose(2, 3))

        scores = scores / (self.head_dim ** 0.5)

        attention = F.softmax(scores, dim=-1)

        context = torch.matmul(attention, values)
        context = context.transpose(1, 2)

        context = context.reshape(batch_size, seq_length, hidden_size)
        output = self.out(context)

        return output

class KANFormer(torch.nn.Module):
    def __init__(self, num_features, hidden_size, num_heads,
                 n_blocks, ff_dims, max_seq_len, device, degree=3):
        super().__init__()

        self.embedding = torch.nn.Linear(num_features, hidden_size)
        head_dim = hidden_size // num_heads
        rope = RotaryPositionalEmbedding(head_dim, max_seq_len)
        rotation_matrix = rope(max_seq_len).to(device)

        self.blocks = torch.nn.ModuleList([
            KANBlock(hidden_size, num_heads, rotation_matrix, degree)
            for _ in range(n_blocks)
        ])

        self.ff = torch.nn.ModuleList()
        
        in_size = max_seq_len * hidden_size
        for f in ff_dims:
            self.ff.append(ChebyshevKANLayer(in_size, f, degree))
            in_size = f

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = x.flatten(start_dim=1)
        for f in self.ff:
            x = f(x) 
        return x

class KANBlock(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, rotation_matrix, degree=3) -> None:
        super().__init__()

        self.norm1 = RMSNorm(hidden_size)
        self.attention = MultiheadKANAttention(hidden_size, num_heads,
                                               rotation_matrix, degree)

    def forward(self, x):
        x1 = self.attention(self.norm1(x))
        out = x + x1
        return out

class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()

        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, dim, max_seq_len):
        super(RotaryPositionalEmbedding, self).__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (1000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.register_buffer('pos_enc', self._generate_positional_encoding(max_seq_len))

    def _generate_positional_encoding(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        pos_enc = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return pos_enc

    def forward(self, seq_len):
        return self.pos_enc[:seq_len, :]

def apply_rotary_pos_emb(x, pos_emb):
    x_cos, x_sin = torch.split(pos_emb, x.shape[-1] // 2, dim=-1)
    x1_rot = (x[..., ::2] * x_cos) + (rotate_half(x[..., 1::2]) * x_sin)
    x2_rot = (x[..., 1::2] * x_cos) + (rotate_half(x[..., ::2]) * x_sin)
    x_rot = torch.cat([x1_rot, x2_rot], dim=-1)
    return x_rot

def rotate_half(x):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)
