import torch
import numpy as np

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def get_batch(data_path, batch_size, context_length, device):
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    ix = np.random.randint(0, len(data) - context_length - 1, size=batch_size)
    x = np.stack([data[i:i+context_length] for i in ix]).astype(np.int64)
    y = np.stack([data[i+1:i+context_length+1] for i in ix]).astype(np.int64)

    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)
    return x, y

