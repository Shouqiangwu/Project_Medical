from dataclasses import dataclass
from typing import Dict, Any
import torch


@dataclass
class Individual:
    # MedVAE latent vectors
    # Shape: [num_classes * IPC, 1, 56, 56]  (for 224x224 input with medvae_4_1_2d)
    z: torch.Tensor

    # Labels for each latent vector
    # Shape: [num_classes * IPC], values in {0,1,2,3,4,5}
    labels: torch.Tensor

    fitness: float = float("-inf")
    eval_info: Dict[str, Any] = None

    def to(self, device):
        self.z = self.z.to(device)
        self.labels = self.labels.to(device)
        return self
