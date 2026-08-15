import sys 
sys.path.insert(0, 'src')

VOCAB_SIZE = 4096
CONTEXT_LENGTH = 256
TARGETS = {'1M': 1_000_000, '4M': 4_000_000, '16M': 16_000_000, '64M': 64_000_000}

def analytic_params(vocab_size, d_model, num_layers, num_heads, d_ff):
    attn = 4 * d_model * d_model
    ffn = 3 * d_model * d_ff
    norms = 2 * d_model
    per_layer = attn + ffn + norms
    non_embed = num_layers * per_layer + d_model
    embed = 2 * vocab_size * d_model
    return embed + non_embed, non_embed

def search(target_non_embed, ratio_lo=4.0, ratio_hi=16.0):
    best = None
    for d_model in [64, 96, 128, 192, 256, 384, 512, 768]:
        for num_heads in [2, 4, 8, 12, 16]:
            if d_model % num_heads != 0:
                continue
            d_ff = ((d_model * 8 // 3) // 64 + 1) * 64
            for num_layers in range(1, 49):
                ratio = d_model / num_layers
                if not (ratio_lo <= ratio <= ratio_hi):
                    continue
                total, non_embed = analytic_params(VOCAB_SIZE, d_model, num_layers, num_heads, d_ff)
                err = abs(non_embed - target_non_embed) / target_non_embed
                if best is None or err < best[0]:
                    best = (err, d_model, num_layers, num_heads, d_ff, total, non_embed)
    return best

if __name__ == '__main__':
    for name, target, in TARGETS.items():
        err, d_model, num_layers, num_heads, d_ff, total, non_embed = search(target)
        print(f'{name}: d_model={d_model} num_layers={num_layers} num_heads={num_heads} d_ff={d_ff} '
              f'| total={total:,} non_embed={non_embed:,} (target {target:,}, err {err:.1%})')

