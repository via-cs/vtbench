import argparse
import yaml
import torch
import os
import shutil
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from vtbench.train.trainer import train_model
from vtbench.train.evaluate import evaluate_model
from vtbench.data.loader import create_dataloaders, read_ucr
from vtbench.models.chart_models.simplecnn import SimpleCNN
from vtbench.models.chart_models.deepcnn import DeepCNN


# CKA UTILITY FUNCTIONS
def compute_cka(X, Y):
    """Compute linear CKA similarity between two feature matrices."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    HSIC = lambda A, B: np.linalg.norm(A.T @ B, 'fro') ** 2
    return HSIC(X, Y) / np.sqrt(HSIC(X, X) * HSIC(Y, Y))

def extract_features(model, dataloader, device):
    """Extract features from the second-to-last layer of the CNN."""
    model.eval()
    feats_all = []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            # Try both methods depending on CNN type
            if hasattr(model, "forward_features"):
                feats = model.forward_features(x)
            elif hasattr(model, "features"):
                feats = model.features(x)
            else:
                feats = model(x)  # fallback (may include classifier layer)
            feats_all.append(feats.cpu().numpy())
    return np.concatenate(feats_all, axis=0)



# EXPERIMENTING WITH CKA

def run_experiment_cka(device):
    """Compute CKA similarity matrices for chart-only models."""
    chart_types = ["line", "area", "bar", "scatter"]
    datasets = {
        "shortlength": ["ECG5000", "Adiac"],
        "mediumlength": ["Strawberry", "Arrowhead"],
        "longlength": ["Beetlefly", "Yoga"]
    }

    os.makedirs("results/cka_heatmaps", exist_ok=True)

    for group_name, dataset_list in datasets.items():
        for dataset_name in dataset_list:
            print(f"\n[CKA] Running CKA experiment for dataset: {dataset_name}")

            chart_features = {}

            for chart_type in chart_types:
                ckpt_path = f"checkpoints/{dataset_name}/{chart_type}_only/best_model.pt"
                if not os.path.exists(ckpt_path):
                    print(f"⚠️ Model not found at {ckpt_path}, skipping {chart_type}.")
                    continue

                # Choose architecture (assuming SimpleCNN for chart-only)
                model = SimpleCNN()
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                model.to(device)

                # Load dataset (using create_dataloaders from vtbench.data.loader)
                train_loader, val_loader, test_loader = create_dataloaders(
                    dataset_name, chart_type, batch_size=32
                )

                # Extract and normalize features
                feats = extract_features(model, test_loader, device)
                feats = StandardScaler().fit_transform(feats)
                chart_features[chart_type] = feats

            # Skip if fewer than 2 chart types available
            if len(chart_features) < 2:
                print(f"Not enough models for {dataset_name} to compute CKA.")
                continue

            # Compute pairwise CKA
            chart_names = list(chart_features.keys())
            n = len(chart_names)
            cka_matrix = np.zeros((n, n))

            for i in range(n):
                for j in range(n):
                    cka_matrix[i, j] = compute_cka(chart_features[chart_names[i]], chart_features[chart_names[j]])

            # Plot and save heatmap
            plt.figure(figsize=(6, 5))
            sns.heatmap(cka_matrix, xticklabels=chart_names, yticklabels=chart_names,
                        annot=True, cmap="coolwarm", vmin=0, vmax=1)
            plt.title(f"CKA Similarity – {dataset_name}")
            plt.tight_layout()
            plt.savefig(f"results/cka_heatmaps/cka_{dataset_name}.png")
            plt.close()

            print(f"[CKA] Heatmap saved for {dataset_name}.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation mode")
    parser.add_argument("--cka", action="store_true", help="Run Experiment 1A: CKA Matrices")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normal training / evaluation workflow
    if args.cka:
        run_experiment_cka(device)

    elif args.evaluate:
        if args.config is None:
            raise ValueError("You must provide a config file for evaluation.")
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        evaluate_model(config, device=device)

    else:
        if args.config is None:
            raise ValueError("You must provide a config file for training.")
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        train_model(config, device=device)
