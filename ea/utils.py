"""
MedVAE loading, latent encoding/decoding, and evolution operators.

CRITICAL API NOTE:
  - Use mvae.encode(img) and mvae.decode(z), NOT mvae.forward(img).
  - mvae.forward() returns a channel-projected latent that CANNOT be decoded back.
  - mvae.encode() returns raw z from VAE posterior sampling that CAN be decoded.
"""
import torch
import torch.nn as nn


# =========================================================
# 1. MedVAE Model Loading
# =========================================================
def load_medvae(device, model_name="medvae_4_1_2d"):
    """
    Load the MedVAE model for chest X-ray generation.

    Args:
        device: torch.device
        model_name: MedVAE variant, default "medvae_4_1_2d"
                    (grayscale, 4x spatial compression, 1 latent channel)
                    For 224x224 input -> latent shape [B, 1, 56, 56]

    Returns:
        mvae: frozen MedVAE model on device
    """
    from medvae import MVAE

    mvae = MVAE(model_name=model_name, modality="xray")
    mvae = mvae.to(device)
    mvae.eval()
    mvae.requires_grad_(False)

    print(f"MedVAE '{model_name}' loaded on {device}")
    return mvae


# =========================================================
# 2. Encode / Decode
# =========================================================
def encode_images(mvae, images, device, batch_size=32):
    """
    Encode real images into MedVAE latent space.

    Args:
        mvae: MedVAE model
        images: [B, 1, 224, 224] in [0, 1]
        device: torch.device
        batch_size: encode in chunks to avoid OOM

    Returns:
        z: [B, 1, 56, 56] latent tensor on device
    """
    # Convert [0,1] -> [-1,1] for MedVAE input
    images_norm = images * 2.0 - 1.0

    all_z = []
    with torch.no_grad():
        for i in range(0, images_norm.shape[0], batch_size):
            batch = images_norm[i : i + batch_size].to(device)
            z = mvae.encode(batch)
            all_z.append(z)

    return torch.cat(all_z, dim=0)


def decode_latents(mvae, z, device, batch_size=32):
    """
    Decode MedVAE latents back to images.

    Args:
        z: [B, 1, 56, 56] latent tensor
        device: torch.device
        batch_size: decode in chunks to avoid OOM

    Returns:
        images: [B, 1, 224, 224] in [0, 1]
    """
    all_imgs = []
    with torch.no_grad():
        for i in range(0, z.shape[0], batch_size):
            batch_z = z[i : i + batch_size].to(device)
            decoded = mvae.decode(batch_z)           # [-1, 1]
            imgs = (decoded + 1.0) * 0.5
            imgs = torch.clamp(imgs, 0, 1)
            all_imgs.append(imgs)

    return torch.cat(all_imgs, dim=0)


def print_latent_stats(z: torch.Tensor, label: str = ""):
    """Print statistics of latent tensor for mutation strength calibration."""
    prefix = f"[{label}] " if label else ""
    print(
        f"{prefix}z stats: "
        f"shape={list(z.shape)}, "
        f"mean={z.mean().item():.4f}, "
        f"std={z.std().item():.4f}, "
        f"min={z.min().item():.4f}, "
        f"max={z.max().item():.4f}"
    )


# =========================================================
# 3. Evolution Operators (shape-agnostic, same as code8)
# =========================================================
def get_current_noise_std(gen: int, total_gen: int, start_std: float = 0.1, end_std: float = 0.01) -> float:
    """Linearly decaying mutation strength."""
    progress = gen / max(1, total_gen - 1)
    return start_std + (end_std - start_std) * progress


def crossover_z(z1: torch.Tensor, z2: torch.Tensor,
                 labels1: torch.Tensor = None, num_classes: int = 6) -> torch.Tensor:
    """
    Class-level crossover: for each class, independently pick from parent1 or parent2.

    This preserves semantic integrity — each class block stays intact,
    and the child combines the best class representations from both parents.

    Args:
        z1, z2: [N_total, 1, 56, 56] latent tensors (ordered by class)
        labels1: [N_total] label tensor to identify class boundaries
        num_classes: number of classes

    Returns:
        child_z: [N_total, 1, 56, 56]
    """
    child_z = z1.clone()

    if labels1 is not None:
        # Use label boundaries for class-level swap
        for c in range(num_classes):
            mask = (labels1 == c)
            if mask.any() and torch.rand(1).item() < 0.5:
                child_z[mask] = z2[mask]
    else:
        # Fallback: assume equal IPC per class, evenly split
        ipc = z1.shape[0] // num_classes
        for c in range(num_classes):
            if torch.rand(1).item() < 0.5:
                start = c * ipc
                end = start + ipc
                child_z[start:end] = z2[start:end]

    return child_z


def mutate_z(z: torch.Tensor, noise_std: float) -> torch.Tensor:
    """Latent space mutation: additive Gaussian noise."""
    noise = torch.randn_like(z) * noise_std
    return z + noise
