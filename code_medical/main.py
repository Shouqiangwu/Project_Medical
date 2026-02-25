"""
Evolutionary Algorithm for Medical Image Dataset Distillation
-------------------------------------------------------------
NIH Chest X-ray 14 (6-class) + MedVAE latent space evolution.

Usage:
  python main.py --data_dir ./data --N 50 --IPC 50 --G 30

Prerequisites:
  1. pip install -r requirements.txt
  2. Download Kaggle 224x224 NIH CXR-14 to --data_dir
  3. Run train_baselines.py first to populate baselines.json
"""
import os
import sys
import argparse
import json
import random

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import torchvision.utils as vutils

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ea.config import EvalCfg
from ea.data import get_nih_train_and_test, build_class_index, NUM_CLASSES
from ea.individual import Individual
from ea.stats import compute_real_stats
from ea.select import select_parents
from ea.utils import (
    load_medvae,
    encode_images,
    decode_latents,
    print_latent_stats,
    get_current_noise_std,
    crossover_z,
    mutate_z,
)
from ea.eval import evaluate_individual
from logger_utils import ExperimentLogger


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print(f"Global Seed set to {seed}")


def main():
    ap = argparse.ArgumentParser(description="EA Medical Image Distillation")
    ap.add_argument("--data_dir", type=str, required=True, help="Path to NIH CXR-14 data")
    ap.add_argument("--medvae_model", type=str, default="medvae_4_1_2d")
    ap.add_argument("--N", type=int, default=50, help="Population size")
    ap.add_argument("--IPC", type=int, default=50, help="Images Per Class")
    ap.add_argument("--G", type=int, default=30, help="Generations")
    ap.add_argument("--mut_start", type=float, default=0.3)
    ap.add_argument("--mut_end", type=float, default=0.05)
    ap.add_argument("--max_per_class", type=int, default=3000, help="Cap per class for training data")
    ap.add_argument("--baselines", type=str, default="baselines.json")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    # 1. Seed
    seed_everything(args.seed)
    device = torch.device(args.device)

    # 2. Logger
    logger = ExperimentLogger(log_dir="logs", filename=f"log_seed{args.seed}.csv")

    # 3. Load baselines
    with open(args.baselines, "r") as f:
        baselines = json.load(f)

    # Check that baselines have been populated
    for mname in ["densenet121", "resnet50", "vit_small"]:
        if baselines["upper"].get(mname, 0.0) == 0.0:
            print(f"WARNING: baselines.json upper bound for {mname} is 0.0!")
            print("  Run train_baselines.py first to populate upper bounds.")

    # 4. Load data
    train_raw, test_norm, mean, std = get_nih_train_and_test(
        args.data_dir, max_per_class=args.max_per_class
    )
    test_loader = torch.utils.data.DataLoader(
        test_norm, batch_size=64, shuffle=False, num_workers=2
    )

    rng = random.Random(args.seed)

    # Compute real data statistics (for distribution matching reward)
    real_stats = compute_real_stats(train_raw, 2000, rng)
    # Move to device
    real_stats = {k: v.to(device) for k, v in real_stats.items()}

    # Build class index for sampling
    class_index = build_class_index(train_raw, num_classes=NUM_CLASSES)

    # 5. Load MedVAE
    mvae = load_medvae(device, model_name=args.medvae_model)

    # 6. Initialize population
    pop = []
    print(f"\nInitializing population (N={args.N}, IPC={args.IPC})...")
    for i in range(args.N):
        ind_z_list = []
        ind_label_list = []

        for c in range(NUM_CLASSES):
            # Sample IPC real images from class c
            pool = class_index[c]
            sample_size = min(args.IPC, len(pool))
            idxs = rng.sample(pool, sample_size)
            real_imgs = torch.stack([train_raw[j][0] for j in idxs], dim=0)
            # real_imgs: [IPC, 1, 224, 224] in [0, 1]

            # Encode to MedVAE latent space
            z = encode_images(mvae, real_imgs, device, batch_size=32)
            lbl = torch.full((sample_size,), c, dtype=torch.long, device=device)

            ind_z_list.append(z)
            ind_label_list.append(lbl)

        final_z = torch.cat(ind_z_list, dim=0)
        final_labels = torch.cat(ind_label_list, dim=0)
        pop.append(Individual(z=final_z, labels=final_labels))

        if i == 0:
            print_latent_stats(final_z, label=f"Individual 0")

    print(f"Population initialized. Each individual: z={pop[0].z.shape}, labels={pop[0].labels.shape}")

    # Suggest mutation strength based on latent statistics
    z_std = pop[0].z.std().item()
    print(f"\nLatent z std = {z_std:.4f}")
    print(f"  Current mut_start={args.mut_start}, mut_end={args.mut_end}")
    print(f"  Suggested mut_start ~ {0.1 * z_std:.4f}, mut_end ~ {0.02 * z_std:.4f}")

    eval_cfg = EvalCfg()

    # 7. Evolution loop
    for gen in range(args.G):
        print(f"\n=== Generation {gen} ===")

        # --- A. Evaluate ---
        for idx, ind in enumerate(pop):
            if ind.fitness == float("-inf"):
                fit = evaluate_individual(
                    ind, mvae, eval_cfg, baselines,
                    test_loader, mean, std, device, real_stats
                )

                info = ind.eval_info
                logger.log_individual(
                    gen=gen, ind_id=idx, fitness=fit,
                    scores_dict=info["per_model_scores"],
                    best_norm_curr=info["best_norm_val"],
                    best_norm_model=info["best_norm_model"],
                )
                print(
                    f"  Ind {idx}: Fit={fit:.4f} | "
                    f"TopScore={info['best_norm_val']:.4f} ({info['best_norm_model']})"
                )

        # Update plot
        logger.plot_curves()

        # Sort by fitness
        pop.sort(key=lambda x: x.fitness, reverse=True)
        best_ind = pop[0]
        print(f"Gen {gen} Best Fitness: {best_ind.fitness:.4f}")

        # Save best images
        with torch.no_grad():
            vis_z = best_ind.z[:64].to(device)
            imgs = decode_latents(mvae, vis_z, device, batch_size=32)
            # Repeat grayscale -> 3 channels for visualization
            imgs_vis = imgs.repeat(1, 3, 1, 1)
            vutils.save_image(
                imgs_vis, f"logs/best_img_gen_{gen}.png", nrow=8, padding=2
            )

        # Save best latent
        torch.save(best_ind.z.cpu(), "best_z.pt")

        if gen == args.G - 1:
            break

        # --- B. Reproduce ---
        elite_size = max(1, int(args.N * 0.2))
        elites = pop[:elite_size]
        parents = select_parents(
            pop, num_pairs=args.N - elite_size, temperature=1.0, rng=rng
        )

        next_gen = []
        current_std = get_current_noise_std(gen, args.G, args.mut_start, args.mut_end)

        for p1, p2 in parents:
            child_z = crossover_z(p1.z, p2.z, labels1=p1.labels, num_classes=NUM_CLASSES)
            child_z = mutate_z(child_z, current_std)
            next_gen.append(Individual(z=child_z, labels=p1.labels.clone()))

        pop = elites + next_gen

    print("\nEvolution Finished.")


if __name__ == "__main__":
    main()
