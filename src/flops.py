def flops_per_forward(vocab_size, context_length, d_model, num_layers, num_heads, d_ff):
    attn_proj = 2 * context_length * 4 * d_model * d_model
    attn_score = 2 * 2 * context_length * context_length * d_model
    ffn = 2 * context_length * 3 * d_model * d_ff
    per_layer = attn_proj + attn_score + ffn
    lm_head = 2 * context_length * d_model * vocab_size
    return num_layers * per_layer + lm_head

def training_flops(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, num_tokens):
    fwd = flops_per_forward(vocab_size, context_length, d_model, num_layers, num_heads, d_ff)
    steps = num_tokens / context_length
    return steps * fwd * 3

if __name__ == '__main__':
    fwd = flops_per_forward(4096, 256, 96, 8, 2, 320)
    print(f'1M config, forward FLOPs per sequence: {fwd:,.0f}')

