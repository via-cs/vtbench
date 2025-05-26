import argparse
import yaml
import torch
from vtbench.train.trainer import train_model
from vtbench.train.evaluate import evaluate_model
from vtbench.data.loader import create_dataloaders, read_ucr
import os
import shutil

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser(description="Run VTBench Experiments")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # ====================
    #  Train the model
    # ====================
    model = train_model(config)

    # ====================
    #  Prepare test loaders and run evaluation
    # ====================
    all_datasets = create_dataloaders(config)
    test_chart = all_datasets['test']['chart']
    test_num = all_datasets['test']['numerical']

    print("\n=== Running Evaluation ===")

    # Use the dispatcher function with explicit model type
    model_type = config['model']['type']
    if model_type == 'single_modal_chart':
        results = evaluate_model(model, test_chart, None, 'single_chart')
    elif model_type == 'two_branch':
        results = evaluate_model(model, test_chart, test_num, 'two_branch')
    elif model_type == 'multi_modal_chart':
        results = evaluate_model(model, test_chart, test_num, 'multi_chart')
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # ====================
    #  Print and save results
    # ====================
    print("\n=== Evaluation Results ===")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric.capitalize()}: {value:.4f}")
        else:
            print(f"{metric.capitalize()}: {value}")

    # Save results
    dataset_name = config['dataset']['name']
    yaml_name = os.path.splitext(os.path.basename(args.config))[0]

    save_dir = os.path.join("results", dataset_name, yaml_name)
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(save_dir, "config_used.yaml"))

    with open(os.path.join(save_dir, 'results.txt'), 'w') as f:
        for metric, value in results.items():
            f.write(f"{metric}: {value}\n")

    print(f"\nResults saved to {save_dir}/results.txt")

if __name__ == '__main__':
    main()