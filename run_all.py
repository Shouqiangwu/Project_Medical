"""
One-click pipeline: Random baseline + EA evolution + Final evaluation.

Runs IPC=1, 10, 50 sequentially. For each IPC:
  1. Random baseline (random real images -> train -> AUROC)
  2. EA evolution (main.py with per-IPC steps)
  3. Final evaluation (eval_final.py with dual seeds 2025/2026)

All outputs organized into logs/ipc{N}/.
Generates summary table + comparison chart at the end.

Usage:
    python run_all.py --data_dir ./data
    python run_all.py --data_dir ./data --ipc 1 10     # only specific IPCs
"""
import os
import sys
import json
import argparse
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ea.data import get_nih_train_and_test, build_class_index, NUM_CLASSES, LABEL_NAMES
from ea.utils import load_medvae, decode_latents
from train_baselines import build_model, set_seed, auroc


# ==================== Config ====================
IPC_STEPS = {
    1:  150,
    10: 200,
    50: 300,
}

IPC_GENERATIONS = {
    1:  15,
    10: 15,
    50: 10,
}

EVAL_SEEDS = (2025, 2026)
EVAL_STEPS = 500        # Final evaluation uses more steps
MODEL_NAMES = ("densenet121", "resnet50", "vit_small")


# ==================== Random Baseline ====================
def evaluate_random_baseline(ipc, train_raw, class_index, test_loader,
                             mean, std, device, out_dir):
    """
    Random baseline: randomly select IPC real images per class, train & evaluate.
    """
    print(f"\n{'='*60}")
    print(f"  RANDOM BASELINE  IPC={ipc}")
    print(f"{'='*60}")

    set_seed(2025)
    rng = random.Random(2025)

    # Sample random real images
    images_list, labels_list = [], []
    for c in range(NUM_CLASSES):
        pool = class_index[c]
        sample_size = min(ipc, len(pool))
        idxs = rng.sample(pool, sample_size)
        imgs = torch.stack([train_raw[j][0] for j in idxs], dim=0)  # [IPC, 1, 224, 224]
        lbls = torch.full((sample_size,), c, dtype=torch.long)
        images_list.append(imgs)
        labels_list.append(lbls)

    train_imgs = torch.cat(images_list, dim=0)   # [IPC*6, 1, 224, 224] in [0,1]
    train_labels = torch.cat(labels_list, dim=0)

    print(f"  Random data shape: {train_imgs.shape}")

    # Evaluate on all models with dual seeds
    results = {}
    for mname in MODEL_NAMES:
        accs = []
        for seed in EVAL_SEEDS:
            set_seed(seed)
            acc = _train_and_auroc(
                mname, train_imgs, train_labels, test_loader,
                mean, std, device, steps=EVAL_STEPS,
            )
            accs.append(acc)
            print(f"  {mname} seed={seed}: AUROC={acc:.4f}")

        mean_acc = np.mean(accs)
        results[mname] = round(mean_acc, 4)
        print(f"  -> {mname} mean AUROC: {mean_acc:.4f}")

    # Save
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "random_results.json"), "w") as f:
        json.dump({"ipc": ipc, "method": "random", "results": results}, f, indent=2)

    return results


def _train_and_auroc(model_name, train_imgs, train_labels, test_loader,
                     mean, std, device, steps=500, lr=0.01, batch_size=256):
    """Train a model on tensor data and return test AUROC."""
    net = build_model(model_name, num_classes=NUM_CLASSES).to(device)
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # Augmentation (no horizontal flip for CXR)
    aug = nn.Sequential(
        T.RandomRotation(degrees=15),
        T.RandomAffine(degrees=0, scale=(0.95, 1.05)),
    )

    mean_t = torch.tensor(mean).view(1, 1, 1, 1).to(device)
    std_t = torch.tensor(std).view(1, 1, 1, 1).to(device)

    imgs = train_imgs.to(device)
    lbls = train_labels.to(device)
    n = imgs.shape[0]

    for step in range(steps):
        idx = torch.randint(0, n, (min(batch_size, n),), device=device)
        x = imgs[idx]
        y = lbls[idx]
        with torch.no_grad():
            x = aug(x)
            x = (x - mean_t) / std_t
        optimizer.zero_grad()
        criterion(net(x), y).backward()
        optimizer.step()

    # Evaluate
    auc_val = auroc(net, test_loader, device, NUM_CLASSES)
    del net, optimizer
    torch.cuda.empty_cache()
    return auc_val


# ==================== EA Evolution ====================
def run_ea(ipc, steps, data_dir, device, N=30, G=30):
    """Run EA evolution via main.py subprocess."""
    out_dir = f"logs/ipc{ipc}"
    cmd = (
        f"{sys.executable} main.py"
        f" --data_dir {data_dir}"
        f" --IPC {ipc}"
        f" --N {N}"
        f" --G {G}"
        f" --steps {steps}"
        f" --out_dir {out_dir}"
        f" --device {device}"
    )
    print(f"\n{'='*60}")
    print(f"  EA EVOLUTION  IPC={ipc}  Steps={steps}  N={N}  G={G}")
    print(f"  Command: {cmd}")
    print(f"{'='*60}")

    ret = os.system(cmd)
    if ret != 0:
        print(f"  [ERROR] EA for IPC={ipc} failed with return code {ret}")
        return False
    return True


# ==================== Final Evaluation ====================
def run_final_eval(ipc, data_dir, device):
    """Run eval_final.py on the best_z.pt from EA."""
    out_dir = f"logs/ipc{ipc}"
    z_path = os.path.join(out_dir, "best_z.pt")

    if not os.path.exists(z_path):
        print(f"  [SKIP] {z_path} not found")
        return None

    print(f"\n{'='*60}")
    print(f"  FINAL EVALUATION  IPC={ipc}")
    print(f"{'='*60}")

    # Load MedVAE and decode
    mvae = load_medvae(torch.device(device))
    data = torch.load(z_path, map_location=device)
    if isinstance(data, dict) and "z" in data:
        zs, labels = data["z"], data["labels"]
    else:
        zs = data
        labels = torch.arange(NUM_CLASSES).repeat_interleave(ipc).long()

    train_imgs = decode_latents(mvae, zs, torch.device(device), batch_size=32).cpu()
    labels = labels.cpu()
    del mvae
    torch.cuda.empty_cache()

    print(f"  Decoded images: {train_imgs.shape}")

    # Load test set
    _, test_norm, mean, std = get_nih_train_and_test(data_dir, max_per_class=None)
    test_loader = DataLoader(test_norm, batch_size=256, shuffle=False, num_workers=4)

    # Evaluate with dual seeds
    results = {}
    for mname in MODEL_NAMES:
        accs = []
        for seed in EVAL_SEEDS:
            set_seed(seed)
            acc = _train_and_auroc(
                mname, train_imgs, labels, test_loader,
                mean, std, device, steps=EVAL_STEPS,
            )
            accs.append(acc)
            print(f"  {mname} seed={seed}: AUROC={acc:.4f}")

        mean_acc = np.mean(accs)
        results[mname] = round(mean_acc, 4)
        print(f"  -> {mname} mean AUROC: {mean_acc:.4f}")

    # Save
    with open(os.path.join(out_dir, "ea_final_results.json"), "w") as f:
        json.dump({"ipc": ipc, "method": "ea", "results": results}, f, indent=2)

    return results


# ==================== Summary & Chart ====================
def generate_summary(all_results, baselines):
    """Print summary table and generate comparison chart."""
    print(f"\n\n{'='*70}")
    print("  MEDICAL DATASET DISTILLATION — FULL RESULTS")
    print(f"{'='*70}")

    # Load upper bounds
    upper = baselines["upper"]

    # Table header
    header = f"  {'Method':<12} {'IPC':>5}"
    for m in MODEL_NAMES:
        header += f"  {m:>14}"
    header += f"  {'Mean':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Full data row
    row = f"  {'Full Data':<12} {'ALL':>5}"
    vals = []
    for m in MODEL_NAMES:
        v = upper.get(m, 0)
        row += f"  {v:>13.4f}"
        vals.append(v)
    row += f"  {np.mean(vals):>7.4f}"
    print(row)
    print("  " + "-" * (len(header) - 2))

    # Per-IPC rows
    table_data = []
    for ipc in sorted(all_results.keys()):
        for method in ["random", "ea"]:
            res = all_results[ipc].get(method)
            if res is None:
                continue
            row = f"  {method.upper():<12} {ipc:>5}"
            vals = []
            for m in MODEL_NAMES:
                v = res.get(m)
                if v is not None:
                    row += f"  {v:>13.4f}"
                    vals.append(v)
                else:
                    row += f"  {'N/A':>14}"
            if vals:
                row += f"  {np.mean(vals):>7.4f}"
            print(row)
            table_data.append({"method": method, "ipc": ipc, "results": res})

    # Save all results
    summary = {
        "baselines": baselines,
        "results": all_results,
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/all_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to logs/all_results.json")

    # Generate chart
    _plot_comparison(all_results, upper)


def _plot_comparison(all_results, upper):
    """Generate bar chart comparing Random vs EA across IPCs."""
    ipcs = sorted(all_results.keys())

    fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(5 * len(MODEL_NAMES), 5))
    if len(MODEL_NAMES) == 1:
        axes = [axes]

    bar_width = 0.3
    x = np.arange(len(ipcs))

    for ax, mname in zip(axes, MODEL_NAMES):
        random_vals = []
        ea_vals = []
        for ipc in ipcs:
            r = all_results.get(ipc, {}).get("random", {}).get(mname, 0)
            e = all_results.get(ipc, {}).get("ea", {}).get(mname, 0)
            random_vals.append(r)
            ea_vals.append(e)

        ax.bar(x - bar_width/2, random_vals, bar_width, label="Random", color="#95a5a6", alpha=0.8)
        ax.bar(x + bar_width/2, ea_vals, bar_width, label="EA (Ours)", color="#e74c3c", alpha=0.8)

        # Full data line
        full_val = upper.get(mname, 0)
        ax.axhline(y=full_val, color="blue", linestyle="--", linewidth=1.5, label=f"Full Data ({full_val:.3f})")

        ax.set_xlabel("IPC")
        ax.set_ylabel("AUROC")
        ax.set_title(mname)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in ipcs])
        ax.set_ylim(0.4, max(full_val + 0.05, 0.85))
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Medical Dataset Distillation: Random vs EA", fontsize=14, fontweight="bold")
    plt.tight_layout()
    chart_path = "logs/medical_comparison_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved to {chart_path}")


# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser(description="Run all medical distillation experiments")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to NIH CXR-14 data")
    parser.add_argument("--ipc", type=int, nargs="+", default=[1, 10, 50], help="IPC values to run")
    parser.add_argument("--N", type=int, default=20, help="EA population size")
    parser.add_argument("--G", type=int, default=30, help="EA generations")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_ea", action="store_true", help="Skip EA, only run random + final eval")
    args = parser.parse_args()

    device = args.device
    start_time = time.time()

    # Load data once
    print("Loading NIH CXR-14 dataset...")
    train_raw, test_norm, mean, std = get_nih_train_and_test(args.data_dir, max_per_class=3000)
    test_loader = DataLoader(test_norm, batch_size=256, shuffle=False, num_workers=4)
    class_index = build_class_index(train_raw, num_classes=NUM_CLASSES)

    # Load baselines
    with open("baselines.json", "r") as f:
        baselines = json.load(f)

    all_results = defaultdict(dict)

    for ipc in args.ipc:
        out_dir = f"logs/ipc{ipc}"
        steps = IPC_STEPS.get(ipc, 200)

        print(f"\n\n{'#'*70}")
        print(f"  IPC = {ipc}  (EA steps={steps})")
        print(f"{'#'*70}")

        # 1. Random baseline
        rand_res = evaluate_random_baseline(
            ipc, train_raw, class_index, test_loader, mean, std, device, out_dir
        )
        all_results[ipc]["random"] = rand_res

        # 2. EA evolution
        if not args.skip_ea:
            G = IPC_GENERATIONS.get(ipc, args.G)
            ok = run_ea(ipc, steps, args.data_dir, device, N=args.N, G=G)

            # 3. Final evaluation
            if ok:
                ea_res = run_final_eval(ipc, args.data_dir, device)
                if ea_res:
                    all_results[ipc]["ea"] = ea_res
        else:
            # Try loading existing EA results
            ea_json = os.path.join(out_dir, "ea_final_results.json")
            if os.path.exists(ea_json):
                with open(ea_json) as f:
                    ea_res = json.load(f)["results"]
                all_results[ipc]["ea"] = ea_res

    # Summary
    generate_summary(dict(all_results), baselines)

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    print(f"\n  Total time: {hours}h {minutes}m")


if __name__ == "__main__":
    main()
