import torch
import numpy as np

def jitter(x, sigma=0.03):
    """加上隨機雜訊"""
    return x + sigma * torch.randn_like(x)

def scaling(x, sigma=0.1):
    """整體縮放"""
    factor = torch.normal(1.0, sigma, size=(x.size(0), 1, 1), device=x.device)
    return x * factor

def masking(x, mask_ratio=0.1):
    """遮蔽部分時間點"""
    B, C, T = x.shape
    mask = torch.ones_like(x)
    num_mask = int(T * mask_ratio)
    for i in range(B):
        idx = np.random.choice(T, num_mask, replace=False)
        mask[i, :, idx] = 0
    return x * mask
