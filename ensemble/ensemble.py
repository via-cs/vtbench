import argparse
import torch
from torch.utils.data import DataLoader
from num.CNN_utils import TimeSeriesImageDatasetMC
from num.data_utils import read_ucr, read_ecg5000
from num.models.SimpleCNN import Simple2DCNN
from num.models.DeepCNN import Deep2DCNN
from num.ensemble_train_test import train_model_ensemble, test_model_ensemble
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, balanced_accuracy_score
from collections import defaultdict
import numpy as np
import os
import yaml
import json


def train_for_config(dataset_config, train_file, test_file, device):
   
    if dataset_config['name'].lower() == "ecg5000":
        X_train, y_train = read_ecg5000(train_file)
        X_test, y_test = read_ecg5000(test_file)
    else:
        X_train, y_train = read_ucr(train_file)
        X_test, y_test = read_ucr(test_file)

    predictions = defaultdict(list)
   
    from torchvision import transforms
    default_transform = transforms.Compose([
        transforms.Resize((64, 64)),  
        transforms.ToTensor(),       
    ])

    results_dir = "ensemble_results"
    os.makedirs(results_dir, exist_ok=True)

    for combo in dataset_config['combos']:
        combo_key = combo['combo_key']
        model_type = combo['model_type']
        result_prefix = os.path.join(results_dir, f"{dataset_config['name']}_{combo_key}")


        pred_file = f"{result_prefix}_preds.npy"
        metrics_file = f"{result_prefix}_metrics.json"

        if os.path.exists(pred_file) and os.path.exists(metrics_file):
            print(f"Loading cached results for {combo_key}")
            raw_preds = np.load(pred_file, allow_pickle=True)
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
        else:
            print(f"Processing combo: {combo_key} with model {model_type}")

            chart_type, *options = combo_key.split('_')
            color_mode = options[1]
            label_mode = options[2]
            scatter_mode = options[0] if chart_type == 'scatter' else None
            bar_mode = options[0] if chart_type == 'bar' else None

            train_dataset = TimeSeriesImageDatasetMC(
                dataset_config['name'], X_train, y_train, split='train',
                chart_type=chart_type, color_mode=color_mode, label_mode=label_mode,
                scatter_mode=scatter_mode, bar_mode=bar_mode, transform=default_transform
            )
            test_dataset = TimeSeriesImageDatasetMC(
                dataset_config['name'], X_test, y_test, split='test',
                chart_type=chart_type, color_mode=color_mode, label_mode=label_mode,
                scatter_mode=scatter_mode, bar_mode=bar_mode, transform=default_transform
            )

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            input_channels = 3  
            num_classes = len(set(y_train))

            model = Deep2DCNN(input_channels, num_classes) if model_type == 'Deep2DCNN' else Simple2DCNN(input_channels, num_classes)
            model.to(device)

            print(f"Training model for combo: {combo_key}")
            train_model_ensemble(model, train_loader, test_loader, num_epochs=20)

            print(f"Evaluating model for combo: {combo_key}")
            raw_preds, metrics = test_model_ensemble(model, test_loader, return_raw=True)

            np.save(pred_file, raw_preds)

            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    metrics[key] = value.tolist()
                elif isinstance(value, (np.float32, np.float64)):
                    metrics[key] = float(value)
                elif isinstance(value, (np.int32, np.int64)):
                    metrics[key] = int(value)

            with open(metrics_file, 'w') as f:
                json.dump(metrics, f)

        predictions[combo_key] = raw_preds

    return predictions

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open("ensemble_config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    for dataset_config in config['datasets']:
        print(f"Processing dataset: {dataset_config['name']}")
        predictions = train_for_config(dataset_config, args.train_file, args.test_file, device)
        
        print(f"Aggregating ensemble predictions for dataset: {dataset_config['name']}")
        ensemble_preds = np.mean(
            [np.vstack(predictions[combo['combo_key']]) for combo in dataset_config['combos']],
            axis=0
        )

        print(f"Evaluating ensemble for dataset: {dataset_config['name']}")
        if dataset_config['name'].lower() == "ecg5000":
            y_test = np.concatenate([read_ecg5000(args.test_file)[1]])
        else:
            y_test = np.concatenate([read_ucr(args.test_file)[1]])

        test_loader = [(None, int(label)) for label in y_test]
        metrics = test_model_ensemble(ensemble_preds, test_loader, return_raw=False)

        # Print metrics
        print(f"Ensemble Metrics for {dataset_config['name']}:")
        print(f"Test Accuracy: {metrics['test_accuracy']:.2f}%")
        print(f"Test Loss: {metrics['test_loss']:.2f}")
        print(f"  Precision: {metrics['precision']:.2f}")
        print(f"  Specificity: {metrics['specificity']:.2f}")
        print(f"  Recall: {metrics['recall']:.2f}")
        print(f"  F1 Score: {metrics['f1_score']:.2f}")
        print(f"  AUC: {metrics['auc']:.2f}")
        print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble model training for time series classification")
    parser.add_argument('--train_file', type=str, required=True, help='Path to the training file')
    parser.add_argument('--test_file', type=str, required=True, help='Path to the testing file')
    args = parser.parse_args()

    main(args)
