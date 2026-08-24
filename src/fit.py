import json
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def power_law(C, a, b, c):
    return a * C**b + c

def load_data(path='results/runs.jsonl'):
    runs = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            runs[r['size']] = r
    return list(runs.values())

def fit_scaling_law(runs):
    C = np.array([r['flops'] for r in runs])
    L = np.array([r['final_val_loss'] for r in runs])

    p0 = [10.0, -0.05, 1.0]
    params, _ = curve_fit(power_law, C, L, p0=p0, maxfev=10000)
    a, b, c = params

    print(f'Fitted: L = {a:.4f} * C^{b:.4f} + {c:.4f}')
    print(f'Scaling exponent b = {b:.4f}')
    print(f'Chinchilla exponent b ≈ -0.050 (from Hoffmann et al. 2022)')
    print(f'Difference: {abs(b - (-0.050)):.4f}')

    L_pred = power_law(C, *params)
    ss_res = np.sum((L - L_pred)**2)
    ss_tot = np.sum((L - np.mean(L))**2)
    print(f'R² = {1 - ss_res/ss_tot:.4f}')

    return params

def plot(runs, params):
    C = np.array([r['flops'] for r in runs])
    L = np.array([r['final_val_loss'] for r in runs])
    labels = [r['size'] for r in runs]

    C_fit = np.logspace(np.log10(C.min()), np.log10(C.max()), 200)
    L_fit = power_law(C_fit, *params)

    plt.figure(figsize=(8, 5))
    plt.loglog(C_fit, L_fit, 'b-', label='Fitted power law', linewidth=2)
    for ci, li, label in zip(C, L, labels):
        plt.loglog(ci, li, 'ro', markersize=8)
        plt.annotate(label, (ci, li), textcoords='offset points', xytext=(8, 4))
    plt.xlabel('Compute (FLOPs)')
    plt.ylabel('Validation Loss')
    plt.title('Scaling Law: Loss vs Compute')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/scaling_law.png', dpi=150)
    print('Plot saved to results/scaling_law.png')

if __name__ == '__main__':
    runs = load_data()
    print(f'Loaded {len(runs)} runs: {[r["size"] for r in runs]}')
    params = fit_scaling_law(runs)
    plot(runs, params)