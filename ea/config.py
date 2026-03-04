from dataclasses import dataclass
from typing import Tuple


@dataclass
class EvalCfg:
    # 3 architecturally diverse models for chest X-ray
    model_names: Tuple[str, str, str] = ("densenet121", "resnet50", "vit_small")

    # Evolution uses single seed for speed; final eval uses dual seeds
    seeds: Tuple[int, ...] = (2025,)

    num_classes: int = 6
    steps: int = 200              # Overridden per-IPC by run_all.py
    batch_size: int = 256         # A100 80GB can handle 224x224 grayscale at this batch size
    train_lr: float = 0.01

    # Robustness score coefficients
    alpha: float = 1.0   # CVaR weight
    beta: float = 1.0    # variance penalty weight
    gamma: float = 0.05  # distribution matching reward weight
