import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import einsum, rearrange


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        std = (2 / (in_features + out_features)) ** 0.5
        weight = torch.empty(out_features, in_features, device=device, dtype=dtype)
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        weight = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
        self.weight = nn.Parameter(weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)
        result = (x / rms) * self.weight
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = self.w1(x)
        gated = F.silu(w1x) * self.w3(x)
        return self.w2(gated)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        k = torch.arange(0, d_k, 2, device=device)
        freqs = 1.0 / (theta ** (k / d_k))
        positions = torch.arange(max_seq_len, device=device)
        angles = torch.outer(positions, freqs)
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos

        x_rot = torch.stack([x1_rot, x2_rot], dim=-1)
        return x_rot.flatten(-2)

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "... seq_q d_k, ... seq_k d_k -> ... seq_q seq_k")
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == False, float("-inf"))

    attn_weights = softmax(scores, dim=-1)
    return einsum(attn_weights, V, "... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v")

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=None, theta=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads ==0, f"d_model={d_model} not divisible by num_heads={num_heads}"
        self.d_k = d_model // num_heads
        self.causal_mask_cache = {}

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope = None
        if max_seq_len is not None and theta is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len,device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        seq_len = x.shape[-2]

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = rearrange(Q, "... seq (heads d_k) -> ... heads seq d_k", heads=self.num_heads)
        K = rearrange(K, "... seq (heads d_k) -> ... heads seq d_k", heads=self.num_heads)
        V = rearrange(V, "... seq (heads d_k) -> ... heads seq d_k", heads=self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        if seq_len not in self.causal_mask_cache:
            self.causal_mask_cache[seq_len] = torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
            )
        causal_mask = self.causal_mask_cache[seq_len]
        attn_output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        attn_output = rearrange(attn_output, "... heads seq d_k -> ... seq (heads d_k)")
        return self.output_proj(attn_output)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len=None, theta=None, device=None, dtype=None):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        y = x + self.attn(self.ln1(x), token_positions)
        z = y + self.ffn(self.ln2(y))
        return z

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, theta, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, max_seq_len=context_length, theta=theta, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(in_indices)
        token_positions = torch.arange(in_indices.shape[-1], device=in_indices.device)
        for layer in self.layers:
            x = layer(x, token_positions)
        return self.lm_head(self.ln_final(x))

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    o_max = inputs.max(dim=-1, keepdim=True).values
    logsumexp = o_max + torch.log(torch.exp(inputs - o_max).sum(dim=-1, keepdim=True))
    targets_logits = inputs.gather(dim=-1, index=targets.unsqueeze(-1))
    losses = logsumexp - targets_logits
    return losses.mean()

