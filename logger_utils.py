import os
import csv
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ExperimentLogger:
    def __init__(self, log_dir="logs", filename="evolution_log.csv"):
        os.makedirs(log_dir, exist_ok=True)
        self.file_path = os.path.join(log_dir, filename)
        self.plot_path = os.path.join(log_dir, "training_curves.png")

        self.headers = [
            "Generation", "Individual_ID", "Fitness",
            "DenseNet121_Score", "ResNet50_Score", "ViTSmall_Score",
            "Best_Norm_Score_Gen", "Best_Norm_Score_Model",
            "Global_Best_Fitness",
        ]

        if not os.path.exists(self.file_path):
            with open(self.file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

        self.global_best_fit = float("-inf")

    def log_individual(self, gen, ind_id, fitness, scores_dict, best_norm_curr, best_norm_model):
        if fitness > self.global_best_fit:
            self.global_best_fit = fitness

        row = [
            gen, ind_id, f"{fitness:.4f}",
            f"{scores_dict.get('densenet121', 0):.4f}",
            f"{scores_dict.get('resnet50', 0):.4f}",
            f"{scores_dict.get('vit_small', 0):.4f}",
            f"{best_norm_curr:.4f}",
            best_norm_model,
            f"{self.global_best_fit:.4f}",
        ]

        with open(self.file_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def plot_curves(self):
        try:
            df = pd.read_csv(self.file_path)
            if len(df) == 0:
                return

            gen_groups = df.groupby("Generation")
            best_fitness_per_gen = gen_groups["Fitness"].max()
            best_norm_per_gen = gen_groups["Best_Norm_Score_Gen"].max()

            plt.figure(figsize=(10, 6))
            plt.plot(
                best_fitness_per_gen.index, best_fitness_per_gen.values,
                label="Max Fitness (Robustness)", marker="o", color="blue",
            )
            plt.plot(
                best_norm_per_gen.index, best_norm_per_gen.values,
                label="Max Normalized Score", marker="s", color="orange", linestyle="--",
            )
            plt.title("Evolution Progress: Fitness & Normalized Score")
            plt.xlabel("Generation")
            plt.ylabel("Score")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.plot_path)
            plt.close()

        except Exception as e:
            print(f"Warning: plotting failed: {e}")
