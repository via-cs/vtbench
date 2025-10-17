# extract_features_and_cka.py

import torch
import yaml
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from vtbench.data.loader import create_dataloaders
from vtbench.models.chart_models.deepcnn import DeepCNN


# ----------------------------------------------
# Centered Kernel Alignment (CKA)
# ----------------------------------------------
def compute_cka(X, Y):
    X -= X.mean(0)
    Y -= Y.mean(0)
    Kx = X @ X.T
    Ky = Y @ Y.T
    hsic = (Kx * Ky).sum()
    var1 = (Kx * Kx).sum() ** 0.5
    var2 = (Ky * Ky).sum() ** 0.5
    return hsic / (var1 * var2 + 1e-8)


# ----------------------------------------------
# Extract penultimate layer features
# ----------------------------------------------
def extract_features(model, dataloader, device):
    model.eval()
    feats = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            x = model.conv_layers(images)
            x = model.feature_extractor(x)
            feats.append(x.cpu().numpy())

    if len(feats) == 0:
        print("No batches found in dataloader — returning empty array.")
        return np.array([])
    return np.concatenate(feats, axis=0)


# ----------------------------------------------
# Run CKA for one dataset (line, area, bar, scatter)
# ----------------------------------------------
def run_cka_for_one_dataset(base_yaml, dataset_name, ckpt_dir="checkpoints"):
    chart_types = ["line", "area", "bar", "scatter"]
    feature_dict = {}

    # Load base YAML
    with open(base_yaml, "r") as f:
        base_config = yaml.safe_load(f)

    for chart in chart_types:
        print(f"\n--- Processing {dataset_name} ({chart}) ---")

        # Modify YAML dynamically
        config = dict(base_config)
        config["chart_branches"]["branch_1"]["chart_type"] = chart
        config["output"]["dir"] = f"Strawberry_results/single_chart/{dataset_name}/{chart}/"
        os.makedirs(config["output"]["dir"], exist_ok=True)

        # Updated path to your checkpoints
        ckpt_path = f"Strawberry_results/single_chart/{chart}/trained_model.pth"
        if not os.path.exists(ckpt_path):
            print(f"Missing checkpoint for {chart}, skipping.")
            continue


        # Create dataloader
        dataloaders = create_dataloaders(config)
        test_loader = dataloaders["test"]["chart"]
        print(f"Test loader batches for {chart}: {len(test_loader)}")

        # Init model
        num_classes = config["model"]["num_classes"]
        input_channels = config["model"]["input_channels"]
        model = DeepCNN(input_channels = input_channels, num_classes=num_classes)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)

        # Extract features
        feats = extract_features(model, test_loader, device)
        if feats.size == 0:
            print(f"No features extracted for {chart}. Skipping.")
            continue

        feature_dict[chart] = feats
        print(f"Extracted features for {dataset_name} – {chart}")

    # If fewer than 2 valid charts → skip
    if len(feature_dict) < 2:
        print(f" Not enough valid charts for {dataset_name}. Skipping CKA heatmap.")
        return

    # Compute CKA matrix
    charts = list(feature_dict.keys())
    n = len(charts)
    cka_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cka_matrix[i, j] = compute_cka(feature_dict[charts[i]], feature_dict[charts[j]])

    # Plot and save
    plt.figure(figsize=(6, 5))
    sns.heatmap(cka_matrix, xticklabels=charts, yticklabels=charts, cmap="viridis", annot=True)
    plt.title(f"CKA Similarity – {dataset_name}")
    os.makedirs("Strawberry_results/cka_heatmaps", exist_ok=True)
    out_path = f"Strawberry_results/cka_heatmaps/{dataset_name}_cka.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"\nSaved heatmap for {dataset_name} → {out_path}")


# ----------------------------------------------
# Entry point
# ----------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_yaml = "/Users/akkumy/Downloads/test_vtbench/vtbench/vtbench/config/single_modal_chart.yaml"
    dataset_name = "Strawberry"
    run_cka_for_one_dataset(base_yaml, dataset_name)
    