import sys
sys.path.insert(0, 'src')
import yaml
import json
import numpy as np
import torch
from model import TransformerLM, cross_entropy
from utils import get_device
from data_utils import get_batch, save_checkpoint
from optimizer import AdamW, get_lr_cosine_schedule, gradient_clipping
from flops import training_flops

def train(size_name, configs_path='configs/sizes.yaml', total_steps=2000,
          warmup_steps=100, batch_size=32, max_lr=3e-4, min_lr=3e-5,
          weight_decay=0.1, max_grad_norm=1.0, eval_interval=200, eval_iters=20,
          log_every=100):
    with open(configs_path) as f:
        cfg = yaml.safe_load(f)

    size_cfg = cfg['sizes'][size_name]
    vocab_size = cfg['vocab_size']
    context_length = cfg['context_length']
    theta = cfg['theta']

    device = get_device()
    train_data = np.memmap('data/train_tokens.bin', dtype=np.uint16, mode='r')
    val_data = np.memmap('data/val_tokens.bin', dtype=np.uint16, mode='r')

    model = TransformerLM(
        vocab_size=vocab_size, context_length=context_length,
        d_model=size_cfg['d_model'], num_layers=size_cfg['num_layers'],
        num_heads=size_cfg['num_heads'], d_ff=size_cfg['d_ff'], theta=theta,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)

    def estimate_loss(data, num_batches):
        model.eval()
        losses = []
        with torch.no_grad():
            for _ in range(num_batches):
                x, y = get_batch(data, batch_size, context_length, device)
                loss = cross_entropy(model(x), y)
                losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    model.train()
    for step in range(total_steps):
        lr = get_lr_cosine_schedule(step, max_lr, min_lr, warmup_steps, total_steps)
        for group in optimizer.param_groups:
            group['lr'] = lr

        x, y = get_batch(train_data, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), max_grad_norm)
        optimizer.step()

        if step % log_every == 0:
            print(f'step {step:5d} | lr {lr:.2e} | train_loss {loss.item():.4f}')

    final_val_loss = estimate_loss(val_data, eval_iters)

    total_tokens = total_steps * batch_size * context_length
    flops = training_flops(vocab_size, context_length, size_cfg['d_model'],
                            size_cfg['num_layers'], size_cfg['num_heads'],
                            size_cfg['d_ff'], total_tokens)

    result = {
        'size': size_name, 'final_train_loss': loss.item(),
        'final_val_loss': final_val_loss, 'total_tokens': total_tokens, 'flops': flops,
    }
    with open('results/runs.jsonl', 'a') as f:
        f.write(json.dumps(result) + '\n')
    print(f'done: {result}')

if __name__ == '__main__':
    train(sys.argv[1])