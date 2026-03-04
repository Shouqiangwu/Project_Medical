# ea/stats.py  --  Distribution statistics (channel-agnostic, works for grayscale)
from typing import Dict
import torch


def _flatten_images(images: torch.Tensor) -> torch.Tensor:
    """
    Flatten to [N, C, H, W] from either:
      - [N, C, H, W]
      - [Classes, IPC, C, H, W]
    """
    if images.dim() == 5:
        C, IPC = images.shape[:2]
        return images.reshape(C * IPC, *images.shape[2:])
    return images


def compute_real_stats(train_raw, sample_n: int, rng) -> Dict[str, torch.Tensor]:
    """
    Compute channel-wise mean/std from real training data.
    For grayscale CXR: returns tensors of shape [1].
    """
    idxs = rng.sample(range(len(train_raw)), min(sample_n, len(train_raw)))
    xs = torch.stack([train_raw[i][0] for i in idxs], dim=0)   # [N, 1, H, W]

    ch_mean = xs.mean(dim=(0, 2, 3))           # [1]
    ch_std = xs.std(dim=(0, 2, 3), unbiased=False)  # [1]

    return {"mean": ch_mean, "std": ch_std}


def dist_match_reward(images: torch.Tensor, real_stats: Dict[str, torch.Tensor]) -> float:
    """
    Distribution matching reward: closer to real data = higher reward.
    """
    images = _flatten_images(images)

    ch_mean = images.mean(dim=(0, 2, 3))
    ch_std = images.std(dim=(0, 2, 3), unbiased=False)

    device = images.device
    real_mean = real_stats["mean"].to(device)
    real_std = real_stats["std"].to(device)

    d_mean = torch.norm(ch_mean - real_mean, p=2)
    d_std = torch.norm(ch_std - real_std, p=2)

    reward = -(d_mean + d_std).item()
    return reward
