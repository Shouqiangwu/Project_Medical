"""
Fitness evaluation for a single Individual.

Flow:
  1. Decode latent z -> synthetic X-ray images via MedVAE
  2. Train 3 eval models (DenseNet-121, ResNet-50, ViT-Small) x 2 seeds
     with class-weighted CrossEntropyLoss
  3. Compute AUROC on test set (not accuracy)
  4. Fitness = alpha * CVaR - beta * variance + gamma * dist_match_reward
"""
import sys
import os

import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ea.config import EvalCfg
from ea.stats import dist_match_reward
from ea.utils import decode_latents
from train_baselines import set_seed, build_model, auroc, compute_class_weights


def evaluate_individual(
    ind, mvae, eval_cfg, baselines, test_loader, mean, std, device, real_stats
) -> float:
    """
    Evaluate an Individual's fitness using multi-model robustness scoring.
    Uses AUROC as the evaluation metric (not accuracy).
    Uses class-weighted CE loss for training to handle imbalance.
    """
    # 1. Decode latents to synthetic images
    synth_images = decode_latents(mvae, ind.z, device, batch_size=32)
    # synth_images: [N, 1, 224, 224] in [0, 1]

    train_imgs = synth_images
    train_labels = ind.labels.to(device)

    # 2. Compute class weights from this individual's label distribution
    labels_list = train_labels.cpu().tolist()
    class_weights = compute_class_weights(labels_list, num_classes=eval_cfg.num_classes)
    class_weights = class_weights.to(device)

    # 3. Data augmentation (NO horizontal flip for chest X-rays)
    aug_transform = nn.Sequential(
        T.RandomRotation(degrees=15),
        T.RandomAffine(degrees=0, scale=(0.95, 1.05)),
    )

    # Normalization tensors for single-channel images
    mean_tensor = torch.tensor(mean).view(1, 1, 1, 1).to(device)
    std_tensor = torch.tensor(std).view(1, 1, 1, 1).to(device)

    # Create DataLoader
    train_ds = TensorDataset(train_imgs, train_labels)
    train_loader = DataLoader(
        train_ds, batch_size=eval_cfg.batch_size, shuffle=True, num_workers=0
    )

    # 4. Multi-model evaluation loop
    baseline_val = baselines["baseline_random_guess"]   # 0.5 for AUC
    upper_map = baselines["upper"]

    all_norm_scores = []
    per_model_scores = {}
    best_norm_val = float("-inf")
    best_norm_model = "None"

    for mname in eval_cfg.model_names:
        model_scores = []
        for seed in eval_cfg.seeds:
            set_seed(seed)
            model = build_model(mname, num_classes=eval_cfg.num_classes).to(device)
            model.train()
            optimizer = torch.optim.SGD(
                model.parameters(), lr=eval_cfg.train_lr, momentum=0.9
            )
            # Weighted loss to handle class imbalance
            criterion = nn.CrossEntropyLoss(weight=class_weights)

            iter_loader = iter(train_loader)

            for _ in range(eval_cfg.steps):
                try:
                    x, y = next(iter_loader)
                except StopIteration:
                    iter_loader = iter(train_loader)
                    x, y = next(iter_loader)

                with torch.no_grad():
                    x = aug_transform(x)
                    x = (x - mean_tensor) / std_tensor

                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

            # Evaluate with AUROC
            auc_val = auroc(model, test_loader, device, eval_cfg.num_classes)

            # Normalize score: (auc - 0.5) / (upper_auc - 0.5)
            upper = upper_map.get(mname, 1.0)
            norm_score = (auc_val - baseline_val) / (upper - baseline_val + 1e-8)

            all_norm_scores.append(norm_score)
            model_scores.append(norm_score)

            if norm_score > best_norm_val:
                best_norm_val = norm_score
                best_norm_model = mname

            del model, optimizer

        per_model_scores[mname] = np.mean(model_scores)

    # 5. Fitness computation
    sorted_scores = sorted(all_norm_scores)
    # CVaR: mean of bottom 2/3 of scores (adapts to number of seeds)
    cvar_k = max(1, int(len(sorted_scores) * 0.67))
    cvar_val = np.mean(sorted_scores[:cvar_k])
    avg_var = np.var(all_norm_scores)

    dm_reward = dist_match_reward(synth_images.detach(), real_stats)

    final_score = (
        eval_cfg.alpha * cvar_val
        - eval_cfg.beta * avg_var
        + eval_cfg.gamma * dm_reward
    )

    ind.fitness = final_score
    ind.eval_info = {
        "cvar": cvar_val,
        "score": final_score,
        "per_model_scores": per_model_scores,
        "best_norm_val": best_norm_val,
        "best_norm_model": best_norm_model,
    }

    del synth_images, train_ds, train_loader
    return final_score
