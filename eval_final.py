"""
Final evaluation of the best evolved latent vectors.

Loads best_z.pt, decodes via MedVAE, and evaluates on multiple models
with data augmentation. Reports per-model accuracy averaged over 3 runs.

Usage:
  python eval_final.py --z_path best_z.pt --data_dir ./data --steps 1000
"""
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ea.data import get_nih_train_and_test, NUM_CLASSES, LABEL_NAMES
from ea.utils import load_medvae, decode_latents
from train_baselines import build_model, set_seed, auroc


# ================= Augmented Dataset =================
class AugmentedTensorDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


# ================= Train + Evaluate One Model =================
def train_one_model(model_name, train_loader, test_loader, device, steps, lr):
    net = build_model(model_name, num_classes=NUM_CLASSES).to(device)
    net.train()

    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    iterator = iter(train_loader)
    for _ in range(steps):
        try:
            inputs, targets = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            inputs, targets = next(iterator)

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Evaluation — use AUROC (standard for medical imaging)
    auc_val = auroc(net, test_loader, device, NUM_CLASSES)
    return auc_val


# ================= Main =================
def main():
    parser = argparse.ArgumentParser(description="Final evaluation of evolved medical data")
    parser.add_argument("--z_path", type=str, default="best_z.pt")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_runs", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load best latent vectors and decode
    if not os.path.exists(args.z_path):
        print(f"Error: {args.z_path} not found")
        return

    print(f"Loading latent vectors from {args.z_path}...")
    mvae = load_medvae(device)

    data = torch.load(args.z_path, map_location=device)
    if isinstance(data, dict) and "z" in data:
        zs, labels = data["z"], data["labels"]
    else:
        zs = data
        ipc = zs.shape[0] // NUM_CLASSES
        labels = torch.arange(NUM_CLASSES).repeat_interleave(ipc).long()

    print(f"Latent shape: {zs.shape}, Labels shape: {labels.shape}")

    # Decode to images
    train_imgs = decode_latents(mvae, zs, device, batch_size=32)
    train_imgs = train_imgs.cpu()
    labels = labels.cpu()
    del mvae
    torch.cuda.empty_cache()

    print(f"Decoded images: {train_imgs.shape}")

    # 2. Prepare datasets
    # Training: augmentation (NO horizontal flip for CXR)
    train_transform = T.Compose([
        T.ToPILImage(),
        T.RandomRotation(15),
        T.RandomAffine(degrees=0, scale=(0.95, 1.05)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ])

    train_ds = AugmentedTensorDataset(train_imgs, labels, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Test set
    _, test_norm, _, _ = get_nih_train_and_test(args.data_dir, max_per_class=None)
    test_loader = DataLoader(test_norm, batch_size=256, shuffle=False, num_workers=4)

    # 3. Evaluate models
    model_list = ["densenet121", "resnet50", "vit_small"]

    print(f"\n{'='*60}")
    print(f"Final Evaluation (Steps={args.steps}, Seeds=2025,2026)")
    print(f"Models: {model_list}")
    print(f"{'='*60}")

    results = {}
    for m in model_list:
        print(f"\nTesting: {m}")
        aucs = []
        for seed in [2025, 2026]:
            set_seed(seed)
            try:
                auc_val = train_one_model(m, train_loader, test_loader, device, args.steps, lr=0.01)
                aucs.append(auc_val)
                print(f"  seed={seed}: AUROC={auc_val:.4f}")
            except Exception as e:
                print(f"  seed={seed} Failed: {e}")

        if aucs:
            mean_auc = np.mean(aucs)
            results[m] = round(mean_auc, 4)
            print(f"  >> {m} mean AUROC: {mean_auc:.4f}")

    # Save results
    import json
    out_path = args.z_path.replace(".pt", "_results.json")
    with open(out_path, "w") as f:
        json.dump({"z_path": args.z_path, "results": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print(f"\n{'='*60}")
    print("Done!")


if __name__ == "__main__":
    main()
