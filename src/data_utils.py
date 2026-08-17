import numpy as np
import torch

def get_batch(dataset, batch_size, context_length, device):
    max_start = len(dataset) - context_length
    starts = np.random.randint(0, max_start, size=batch_size)

    inputs = np.stack([dataset[s : s + context_length] for s in starts])
    targets = np.stack([dataset[s + 1 : s + context_length + 1] for s in starts])

    inputs = torch.from_numpy(inputs).long().to(device)
    targets = torch.from_numpy(targets).long().to(device)
    return inputs, targets

def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)

def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    try:
        optimizer.load_state_dict(checkpoint["optimizer"])
    except (KeyError, ValueError, RuntimeError):
        print('optimizer state incompatible, skipping (momentum will rebuild)')
    return checkpoint["iteration"]