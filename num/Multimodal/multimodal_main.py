import torch
import logging
import argparse
import yaml
import numpy as np
import random
from pathlib import Path
import torch.nn as nn

from multimodalCNN import MultimodalDeep2DCNN
from multimodal_train import create_dataloaders, train_model, evaluate_model
from mm_data_utils import read_ucr, normalize_data, to_torch_tensors



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

    numerical_train = X_train
    numerical_test = X_test
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        X_train=X_train,
        numerical_train=numerical_train,
        y_train=y_train,
        X_test=X_test,
        numerical_test=numerical_test,
        y_test=y_test,
        train_file=config['train_file'],
        batch_size=32
    )

    first_key = next(iter(dataloaders.keys()))
    print(f"Using dataloaders for key: {first_key}")
    train_loader, val_loader, test_loader = dataloaders[first_key]
    
    model = MultimodalDeep2DCNN(
        input_channels=3,  
        num_classes=config['num_classes'],  
        num_numerical_features=numerical_train.shape[1]  
    ).to(device)
    

    trained_model, val_loss, best_val_accuracy = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        patience=config['patience']
    )


    test_metrics = evaluate_model(
        model=trained_model,
        test_loader = test_loader
    )

if __name__ == '__main__':
    main()
