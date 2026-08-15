import sys
sys.path.insert(0, 'src')
import torch
from model import TransformerLM

def count_params(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, theta=10000.0):
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        theta=theta,
    )
    total = sum(p.numel() for p in model.parameters())
    embed = model.token_embeddings.weight.numel() + model.lm_head.weight.numel()
    return total, total - embed

if __name__ == '__main__':
    total, non_embed = count_params(
        vocab_size=4096, context_length=256,
        d_model=64, num_layers=2, num_heads=4, d_ff=192,
    )
    print(f'total params: {total:,}')
    print(f'non-embedding params: {non_embed:,}')

