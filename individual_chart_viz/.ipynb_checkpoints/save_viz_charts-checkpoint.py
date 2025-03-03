import numpy as np
import torch
import json
import yaml
import os
import sys
from sklearn.metrics import confusion_matrix

from num.data_utils import read_ucr, read_ecg5000, normalize_data, to_torch_tensors
from num.models.SimpleCNN import Simple2DCNN  
from num.models.DeepCNN import Deep2DCNN 
from num.CNN_train import create_dataloaders, train_model


def load_config(config_path):
    """Load the YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def train_and_evaluate_best_model(combo_key, model_type, X_train, y_train, X_test, y_test, train_file, dataset_name, best_models_results_dict):
    """Train and evaluate the model for a given combo."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train, y_train, X_test, y_test = to_torch_tensors(X_train, y_train, X_test, y_test)

    dataloaders = create_dataloaders(X_train, y_train, X_test, y_test, train_file, dataset_name,verbose=False)
    if combo_key not in dataloaders:
        print(f"Combination {combo_key} not found in dataloaders.")
        return

    train_loader, val_loader, test_loader = dataloaders[combo_key]

    model = model_type(3, len(torch.unique(y_train))).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    num_epochs = 100
    train_model(model, train_loader, val_loader, num_epochs, patience=10, optimizer=optimizer)

    model.eval()
    y_true, y_pred = [], []
    with torch.inference_mode():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    y_true = list(map(int, y_true))
    y_pred = list(map(int, y_pred))
    best_models_results_dict[combo_key] = {
        'true_labels': y_true,
        'predicted_labels': y_pred
    }

    conf_matrix = confusion_matrix(y_true, y_pred)
    print(f'Confusion Matrix for {combo_key}:')
    print(conf_matrix)


def main(config_path):
    """Main function to load config, train models, and save results."""
    
    config = load_config(config_path)
    best_models_results_dict = {}

    for dataset in config['datasets']:
        dataset_name = dataset['name']
        train_file = dataset['train_file']
        test_file = dataset['test_file']

        if dataset_name.lower() == "ecg5000":
            X_train, y_train = read_ecg5000(train_file)
            X_test, y_test = read_ecg5000(test_file)
        else:
            X_train, y_train = read_ucr(train_file)
            X_test, y_test = read_ucr(test_file)
        
        X_train, X_test = normalize_data(X_train, X_test)  # Normalize all datasets

        for combo in dataset['combos']:
            combo_key = combo['combo_key']
            model_type = eval(combo['model_type']) 
            print(f"Processing combo: {combo_key} for dataset: {dataset_name}")

            train_and_evaluate_best_model(combo_key, model_type, X_train, y_train, X_test, y_test, train_file, dataset_name, best_models_results_dict)


        json_file_name = (f"{dataset_name}_results.json")
        with open(json_file_name, 'w') as f:
            json.dump(best_models_results_dict, f, indent=4)
        print(f"Results for {dataset_name} saved to '{json_file_name}'.")


if __name__ == "__main__":
    config_path = "config_viz.yaml"  
    main(config_path)