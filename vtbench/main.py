import argparse
import yaml
import torch
from vtbench.train.trainer import train_model
from vtbench.train.evaluate import evaluate_model
from vtbench.data.loader import create_dataloaders, read_ucr
import os

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
    #  Prepare test loaders
    # ====================
    test_datasets = create_dataloaders(config, split='test')

    if config['model']['type'] == 'single_modal_chart':
        test_loader = torch.utils.data.DataLoader(
            test_datasets[0], batch_size=config['training']['batch_size'], shuffle=False
        )
        test_num_loader = None

    else:
        test_loader = [torch.utils.data.DataLoader(
            dataset, batch_size=config['training']['batch_size'], shuffle=False
        ) for dataset in test_datasets]

        # Only load numerical if needed
        if config['model']['type'] in ['two_branch', 'multi_modal_chart_numerical']:
            from vtbench.data.loader import NumericalDataset
            X_test, y_test = read_ucr(config['dataset']['test_path'])
            test_num_loader = torch.utils.data.DataLoader(
                NumericalDataset(X_test, y_test),
                batch_size=config['training']['batch_size'],
                shuffle=False
            )
        else:
            test_num_loader = None

    # ====================
    #  Evaluation
    # ====================
    print("\n=== Running Evaluation ===")

    if config['model']['type'] in ['single_modal_chart', 'multi_modal_chart']:
        results = evaluate_model(model, test_loader)
    elif config['model']['type'] in ['two_branch', 'multi_modal_chart_numerical']:
        results = evaluate_model(model, test_loader, test_num_loader)
    else:
        raise ValueError(f"Unsupported model type: {config['model']['type']}")

    # ====================
    #  Save results
    # ====================
    save_dir = config['output']['dir']
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'results.txt'), 'w') as f:
        for metric, value in results.items():
            f.write(f"{metric}: {value}\n")

    print(f"\nResults saved to {save_dir}/results.txt")

if __name__ == '__main__':
    main()
