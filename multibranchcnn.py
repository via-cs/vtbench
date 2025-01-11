import torch
import torch.nn as nn
import torch.optim as optim
import inspect
import yaml
import numpy as np
from vtbench.JointCNN_train import train_joint_cnn, test_model, create_dataloaders
from vtbench.models.MultiBranchCNN import MultiBranchCNN
from vtbench.data_utils import read_ucr, normalize_data, to_torch_tensors, apply_smote
from torch.nn import CrossEntropyLoss

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_config(config_path="config.yaml"):
    """Load the configuration from config.yaml."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Load configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    combo_keys = [model['combo_key'] for model in config['models']]
    num_classes = config.get('num_classes', 5)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_file = 'data/ECG5000/ECG5000_TRAIN.ts'
    test_file = 'data/ECG5000/ECG5000_TEST.ts'
    x_train, y_train = read_ucr(train_file)
    x_test, y_test = read_ucr(test_file)
    
    # Create model and move to device
    model = MultiBranchCNN(num_classes=num_classes)
    model = model.to(device)
    
    # Create dataloaders
    dataloaders = create_dataloaders(x_train, y_train, x_test, y_test, config)
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01,
        amsgrad=True
    )
    
    # Initialize scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=config['num_epochs'],
        steps_per_epoch=len(dataloaders[combo_keys[0]]['train']),
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0
    )
    
    # Setup criterion with class weights
    class_weights = torch.tensor([1.0, 1.2, 2.0, 1.0, 3.0]).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.1
    )
    
    print("\nStarting training...")
    model, _ = train_joint_cnn(  # Note the unpacking of both return values
        model=model,
        dataloaders=dataloaders,
        combo_keys=combo_keys,
        num_epochs=config['num_epochs'],
        patience=15,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    print("\nStarting testing phase...")
    test_results = test_model(
        model=model,
        test_dataloaders=dataloaders,
        criterion=criterion,
        combo_keys=combo_keys,
        device=device
    )
    
    # Save results
    results = {
        'model_state': model.state_dict(),
        'test_results': test_results,
        'config': config
    }
    torch.save(results, 'model_results.pt')

if __name__ == "__main__":
    main()