import math
import torch

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                grad = p.grad.data

                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad ** 2)

                alpha_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                p.data -= lr * weight_decay * p.data
                p.data -= alpha_t * m / (torch.sqrt(v) + eps)

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss

def get_lr_cosine_schedule(t, alpha_max, alpha_min, warmup_iters, cosine_cycle_iters):
    if t < warmup_iters:
        return (t / warmup_iters) * alpha_max
    elif t <= cosine_cycle_iters:
        progress = (t - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return alpha_min + 0.5 * (1 + math.cos(progress * math.pi)) * (alpha_max - alpha_min)
    else:
        return alpha_min

def gradient_clipping(parameters, max_l2_norm, eps=1e-6):
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm_sq = sum((g ** 2).sum() for g in grads)
    total_norm = total_norm_sq ** 0.5

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)


