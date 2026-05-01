import torch
import yaml
import os
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from data.loader import create_dataloaders
from models.chart_models.deepcnn import DeepCNN  


# Centered Kernel Alignment (CKA)
def compute_cka(X, Y):
    X -= X.mean(0)
    Y -= Y.mean(0)
    Kx = X @ X.T
    Ky = Y @ Y.T
    hsic = (Kx * Ky).sum()
    var1 = (Kx * Kx).sum() ** 0.5
    var2 = (Ky * Ky).sum() ** 0.5
    return hsic / (var1 * var2 + 1e-8)



# skips mismatched keys
def safe_load_model(model, ckpt_path, device="cpu"):
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_dict = model.state_dict()

    filtered = {
        k: v for k, v in checkpoint.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"Loaded pretrained weights from {os.path.basename(ckpt_path)} (skipped classifier (mismatched))")
    if missing:
        print(f"Missing keys (ignored): {missing}")
    if unexpected:
        print(f"Unexpected keys (ignored): {unexpected}")
    return model



# Extract penultimate layer features
def extract_features(model, dataloader, device):
    model.eval()
    feats = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            x = model.conv_layers(images)
            x = model.feature_extractor(x)
            feats.append(x.cpu().numpy())

    if not feats:
        print("No batches found in dataloader — returning empty array.")
        return np.array([])
    return np.concatenate(feats, axis=0)



# Run CKA for one dataset (line, area, bar, scatter)
def run_cka_for_one_dataset(base, dataset_name, device):
    chart_types = ["line", "area", "bar", "scatter"]
    feature_dict = {}

    for chart in chart_types:
        base_yaml = f"{base}/{dataset_name}_{chart}.yaml"
        with open(base_yaml, "r") as f:
            base_config = yaml.safe_load(f)

        print(f"\n--- Processing {dataset_name} ({chart}) ---")

        # Modify YAML dynamically
        config = dict(base_config)
        config["chart_branches"]["branch_1"]["chart_type"] = chart
        output_dir = f"{dataset_name}_results/single_chart/{chart}/"
        config["output"]["dir"] = output_dir
        os.makedirs(output_dir, exist_ok=True)

        ckpt_path = os.path.join(output_dir, "trained_model.pth")
        if not os.path.exists(ckpt_path):
            print(f"Missing checkpoint for {chart}")
            continue

        # Initialize model based on YAML config
        model_type = config["model"]["chart_model"]
        input_channels = config["model"].get("input_channels", 3)
        num_classes = config["model"].get("num_classes", 2)

        if model_type.lower() == "deepcnn":
            model = DeepCNN(input_channels=input_channels, num_classes=num_classes)
        else:
            raise ValueError(f"Model type {model_type} not supported yet")

        # Load weights
        model = safe_load_model(model, ckpt_path, device)
        model.to(device)

        # Create dataloaders
        dataloaders = create_dataloaders(config)
        test_loader = dataloaders["test"]["chart"]
        print(f"Test loader batches for {chart}: {len(test_loader)}")

        # Extract features
        feats = extract_features(model, test_loader, device)
        if feats.size == 0:
            print(f"No features extracted for {chart}")
            continue

        # Save features for probe
        chart_dir = os.path.join(f"{dataset_name}_results", "single_chart", chart)
        os.makedirs(chart_dir, exist_ok=True)

        # Save features + corresponding labels
        np.save(os.path.join(chart_dir, "features.npy"), feats)
        all_labels = []
        for _, labels in test_loader:
            all_labels.extend(labels.cpu().numpy())
        np.save(os.path.join(chart_dir, "labels.npy"), np.array(all_labels))

        feature_dict[chart] = feats
        print(f"Saved features and labels for {chart}")


    # Compute and Save CKA 
    if len(feature_dict) < 2:
        print(f"Not enough valid charts for {dataset_name}")
        return

    charts = list(feature_dict.keys())
    n = len(charts)
    cka_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cka_matrix[i, j] = compute_cka(feature_dict[charts[i]], feature_dict[charts[j]])
            
    # Future experiment step: Compute ESI_CKA before plotting
    esi_cka = 1 - mean_offdiag(cka_matrix)
    print(f"ESI_CKA={esi_cka:.6f}") 

    plt.figure(figsize=(6, 5))
    sns.heatmap(cka_matrix, xticklabels=charts, yticklabels=charts, cmap="viridis", annot=True)
    plt.title(f"CKA Similarity – {dataset_name}")

    out_dir = f"{dataset_name}_results/cka_heatmaps"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_name}_cka.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"\nSaved heatmap for {dataset_name} → {out_path}")


def mean_offdiag(matrix):
    """Compute mean of off-diagonal elements."""
    arr = np.array(matrix)
    n = arr.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return arr[mask].mean()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--output", type=str, required=True, help="Name of the dataset")
    args = parser.parse_args()
    dataset_name = args.dataset
    base_yaml = "/vtbench/vtbench/config/generated_configs"
    run_cka_for_one_dataset(base_yaml, dataset_name, device)
