import os
import torch
import numpy as np
from tqdm import tqdm
import yaml

from models.chart_models.deepcnn import DeepCNN
from models.chart_models.simplecnn import SimpleCNN
from data.loader import create_dataloaders

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_model(config, device):
    model_type = config["model"]["chart_model"]  # e.g., "DeepCNN" or "SimpleCNN"
    if "deep" in model_type.lower():
        model = DeepCNN(input_channels = config["model"]["input_channels"], num_classes=config["model"]["num_classes"])
    else:
        model = SimpleCNN(input_channels = config["model"]["input_channels"], num_classes=config["model"]["num_classes"])
    return model.to(device)

@torch.no_grad()
def extract_features(model, dataloader, device):
    model.eval()
    features, labels = [], []

    for x, y in tqdm(dataloader, desc="Extracting features"):
        x = x.to(device)
        feats = model.forward_features(x) if hasattr(model, "forward_features") else model(x)
        features.append(feats.cpu().numpy())
        labels.append(y.numpy())

    return np.concatenate(features), np.concatenate(labels)

def main():
    config_path = "config/single_modal_chart.yaml"
    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_name = config["dataset"]["name"]
    base_dir = f"{dataset_name}_results/single_chart"
    chart_types = ["line", "area", "bar", "scatter"]

    dataloaders = create_dataloaders(config)
    test_loader = dataloaders["test"]["chart"]

    for chart_type in chart_types:
        ckpt_path = os.path.join(base_dir, chart_type, "trained_model.pth")
        if not os.path.exists(ckpt_path):
            print(f"Missing checkpoint for {chart_type}, skipping.")
            continue

        print(f"\n🔹 Processing {chart_type}...")
        model = get_model(config, device)
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)

        X, y = extract_features(model, test_loader, device)

        np.save(os.path.join(base_dir, chart_type, "features.npy"), X)
        np.save(os.path.join(base_dir, chart_type, "labels.npy"), y)
        print(f"Saved features for {chart_type}: {X.shape}")

if __name__ == "__main__":
    main()
