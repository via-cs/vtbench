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


def load_config(config_path: str):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_output_dir(config):
    """Ensure output directory exists."""
    output_dir = config['output'].get('dir', 'results/')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def main():
    # ========================
    # Load configuration
    # ========================
    config_path = "config/single_modal_chart.yaml"  # Change this if your file has a different name
    config = load_config(config_path)

    # ========================
    # Setup and sanity checks
    # ========================
    output_dir = setup_output_dir(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Training model type: {config['model']['type']}")
    print(f"Results will be saved to: {output_dir}")

    # ========================
    # Train the model
    # ========================
    model = train_model(config)

    # ========================
    # Save trained model
    # ========================
    if config['output'].get('save_model', False):
        model_save_path = os.path.join(output_dir, "trained_model.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to: {model_save_path}")

    print("Training complete!")

if __name__ == "__main__":
    main()

