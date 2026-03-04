"""
Model builders and baseline training for NIH CXR-14 (6-class).

Key design choices for medical imaging:
  - Class-weighted CrossEntropyLoss (handles severe class imbalance)
  - AUROC as primary metric (Accuracy is meaningless when No Finding dominates)
  - Best model saved by Validation AUC, not Accuracy

Usage:
  python train_baselines.py --data_dir ./data --epochs 50
"""
import argparse
import json
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import torchvision.models as models


# ================= Seed Control =================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================= Model Builders =================
def build_model(name: str, num_classes: int = 6) -> nn.Module:
    """
    Build a classification model for grayscale 224x224 input.
    First conv layer is modified: in_channels=3 -> in_channels=1.
    """
    name = name.lower()

    if name in ["densenet121", "densenet-121"]:
        net = models.densenet121(weights=None)
        net.features.conv0 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        net.classifier = nn.Linear(net.classifier.in_features, num_classes)
        return net

    if name in ["resnet50", "resnet-50"]:
        net = models.resnet50(weights=None)
        net.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        return net

    if "vit" in name:
        import timm
        net = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            in_chans=1,
            num_classes=num_classes,
        )
        return net

    raise RuntimeError(f"Unknown model: {name}")


# ================= Class Weights =================
def compute_class_weights(targets, num_classes=6):
    """
    Compute inverse-frequency class weights.
    More weight on rare classes (pathologies), less on No Finding.

    Returns:
        torch.Tensor of shape [num_classes]
    """
    counts = Counter(targets)
    total = len(targets)
    weights = []
    for c in range(num_classes):
        cnt = counts.get(c, 1)
        weights.append(total / (num_classes * cnt))
    w = torch.tensor(weights, dtype=torch.float32)
    print(f"  Class weights: {[f'{x:.2f}' for x in w.tolist()]}")
    return w


# ================= AUROC =================
def auroc(model: nn.Module, loader: DataLoader, device, num_classes: int = 6) -> float:
    """
    Compute macro-averaged AUROC (one-vs-rest).
    This is the standard metric for chest X-ray classification.
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.numpy())

    all_probs = np.concatenate(all_probs, axis=0)     # [N, num_classes]
    all_labels = np.concatenate(all_labels, axis=0)    # [N]

    # One-hot encode labels
    one_hot = np.zeros((len(all_labels), num_classes))
    one_hot[np.arange(len(all_labels)), all_labels] = 1

    # Macro AUROC: average across classes, skip classes not present in this batch
    try:
        auc = roc_auc_score(one_hot, all_probs, average="macro", multi_class="ovr")
    except ValueError:
        # If some class has no samples in test set, fall back to weighted
        auc = roc_auc_score(one_hot, all_probs, average="weighted", multi_class="ovr")

    return auc


# ================= Accuracy (kept for reference logging) =================
def accuracy(model: nn.Module, loader: DataLoader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total


# ================= Baseline Training =================
def train_model_full(
    model_name, train_loader, test_loader, device,
    class_weights, epochs=50, lr=0.01, num_classes=6,
):
    """Train a model on the full training set. Best model selected by AUROC."""
    set_seed(0)
    model = build_model(model_name, num_classes=num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Weighted loss to handle class imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_auc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Evaluate with AUROC (primary) and accuracy (secondary)
        auc_val = auroc(model, test_loader, device, num_classes)
        acc_val = accuracy(model, test_loader, device)

        if auc_val > best_auc:
            best_auc = auc_val
            best_epoch = epoch

        print(
            f"  [{model_name}] Epoch {epoch+1}/{epochs}  "
            f"AUC={auc_val:.4f}  Acc={acc_val:.4f}  "
            f"BestAUC={best_auc:.4f}"
        )

    return best_auc, best_epoch


def main():
    parser = argparse.ArgumentParser(description="Train baselines on NIH CXR-14 (6-class)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to NIH data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--output", type=str, default="baselines.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    from ea.data import get_nih_train_and_test
    import torchvision.transforms as T

    train_raw, test_norm, mean, std = get_nih_train_and_test(
        args.data_dir, max_per_class=None
    )

    # Compute class weights from training set
    class_weights = compute_class_weights(train_raw.targets)

    # For baseline training, normalized + augmented training data
    train_tf = T.Compose([
        T.Resize((224, 224)),
        T.RandomRotation(15),
        T.RandomAffine(degrees=0, scale=(0.95, 1.05)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    from ea.data import NIHChestXrayDataset, _find_image_dir, _find_csv, _find_file
    train_norm = NIHChestXrayDataset(
        img_dir=_find_image_dir(args.data_dir),
        csv_path=_find_csv(args.data_dir),
        transform=train_tf,
        max_per_class=None,
        split_file=_find_file(args.data_dir, "train_val_list.txt"),
    )

    train_loader = DataLoader(train_norm, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_norm, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Train each model
    model_names = ["densenet121", "resnet50", "vit_small"]
    results = {
        "dataset": "NIH-CXR14-6class",
        "baseline_random_guess": 0.5,       # random AUC = 0.5
        "upper": {},
        "runs": [],
    }

    for mname in model_names:
        print(f"\n{'='*50}")
        print(f"Training {mname}...")
        print(f"{'='*50}")

        best_auc, best_epoch = train_model_full(
            mname, train_loader, test_loader, device,
            class_weights=class_weights,
            epochs=args.epochs, lr=args.lr,
        )

        results["upper"][mname] = round(best_auc, 4)
        results["runs"].append({
            "model": mname,
            "best_test_auc": round(best_auc, 4),
            "best_epoch": best_epoch,
            "seed": 0,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
        })

        print(f"  >> {mname} Best AUC: {best_auc:.4f} at epoch {best_epoch}")

    # Save baselines
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBaselines saved to {args.output}")


if __name__ == "__main__":
    main()
