import torch
import logging
import argparse
import yaml
import numpy as np
import random
from pathlib import Path
import torch.nn as nn

from num.models.multibranchCNN import MultiBranchCNN
from num.multibranch_train import create_dataloaders, train, test
from num.data_utils import read_ucr, normalize_data, to_torch_tensors



def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description='Train MultiBranch CNN for Time Series Classification')
    parser.add_argument('--train_path', type=str, required=True, help='Path to training dataset')
    parser.add_argument('--test_path', type=str, required=True, help='Path to testing dataset')
    args = parser.parse_args()

    setup_logging()
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Update config with dataset paths from arguments
    config['train_file'] = args.train_path
    config['test_file'] = args.test_path
    
    # Load and preprocess data
    x_train, y_train = read_ucr(config['train_file'])
    x_test, y_test = read_ucr(config['test_file'])
    x_train, x_test = normalize_data(x_train, x_test)
    X_train, y_train, X_test, y_test = to_torch_tensors(x_train, y_train, x_test, y_test)
    
    # Create dataloaders
    dataloaders = create_dataloaders(X_train, y_train, X_test, y_test, config)
    
    # Create model
    model = MultiBranchCNN(
        input_channels=3,
        num_classes=config['num_classes']
    ).to(device)
    
    # Train model (includes validation)
    trained_model = train(
        model=model,
        dataloaders=dataloaders,
        config=config,
        device=device,
        train_labels = y_train
    )

    # Test model
    test_metrics = test(
        model=trained_model,
        dataloaders=dataloaders,
        device=device
    )

if __name__ == '__main__':
    main()
