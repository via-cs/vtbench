import pandas as pd
import torch
import numpy as np
import traceback
import os
import argparse
from collections import Counter

from num.data_utils import read_ucr, read_ecg5000, normalize_data
from num.models.SimpleCNN import Simple2DCNN
from num.models.DeepCNN import Deep2DCNN
from num.CNN_train import create_dataloaders, train_model, evaluate_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run CNN on various datasets with different configurations.")
    parser.add_argument('--train_file', type=str, required=True, help='Path to the training dataset')
    parser.add_argument('--test_file', type=str, required=True, help='Path to the testing dataset')
    parser.add_argument('--augment', action="store_true", help="Enable image augmentation")
    return parser.parse_args()

def get_dataset_name_from_path(dataset_path):
    return os.path.basename(os.path.dirname(dataset_path))

def get_unique_filename(dataset_name):
    base_name = f"{dataset_name}_results.xlsx"
    if not os.path.exists(base_name):
        return base_name
    counter = 1
    while os.path.exists(f"{dataset_name}_results_{counter}.xlsx"):
        counter += 1
    return f"{dataset_name}_results_{counter}.xlsx"

def write_results_to_excel(file_path, results, version):
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        df = pd.DataFrame()
    results_df = pd.DataFrame([results])
    results_df['Version'] = version
    df = pd.concat([df, results_df], ignore_index=True)
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
        df.to_excel(writer, index=False)

def calculate_and_save_averages(file_path, model_name, combo_key):
    print(f"\nCalculating averages for model '{model_name}' and combination '{combo_key}'...\n")
    df = pd.read_excel(file_path)
    average_results = df.mean(numeric_only=True).to_dict()
    average_results.update({'Model': model_name, 'Combination': combo_key, 'Version': 'Average'})
    df = pd.concat([df, pd.DataFrame([average_results])], ignore_index=True)
    df.to_excel(file_path, index=False)
    print(f"Averages saved to: {file_path}")

def main():
    args = parse_arguments()
    print(f"Train File: {args.train_file}, Test File: {args.test_file}")
    dataset_name = get_dataset_name_from_path(args.train_file)
    excel_file_path = get_unique_filename(dataset_name)
    print(f"Results will be saved to: {excel_file_path}")

    try:
        if "ECG5000" in dataset_name:
            print("Detected ECG5000 dataset. Using `read_ecg5000()`...")
            x_train, y_train = read_ecg5000(args.train_file)
            x_test, y_test = read_ecg5000(args.test_file)
            label_map = None
        else:
            print("Using `read_ucr()` for dataset:", dataset_name)
            x_train, y_train, label_map = read_ucr(args.train_file)
            x_test, y_test, _ = read_ucr(args.test_file)

        print(f"Data loaded successfully. Train samples: {len(y_train)}, Test samples: {len(y_test)}")
        print(f"Train class distribution: {Counter(y_train)}")

    except Exception as e:
        print(f"Error loading data: {e}")
        traceback.print_exc()
        return

    nb_classes = len(np.unique(y_train))
    x_train, x_test = normalize_data(x_train, x_test)

    model_configurations = {
        'Simple2DCNN': Simple2DCNN,
        'Deep2DCNN': Deep2DCNN
    }

    dataloaders = create_dataloaders(
        x_train, y_train, x_test, y_test,
        train_file=args.train_file,
        dataset_name=dataset_name,
        augment=args.augment
    )

    print("Dataloaders created successfully.")

    for model_name, model_class in model_configurations.items():
        print(f"Using model: {model_name}")
        for combo_key, loaders in dataloaders.items():
            print(f"Processing combination: {combo_key}")
            train_loader, val_loader, test_loader = loaders

            for iteration in range(10):
                print(f"Running iteration {iteration + 1} for {model_name} with {combo_key}")
                model = model_class(3, nb_classes).to(device)
                best_val_loss, best_val_accuracy = train_model(model, train_loader, val_loader, num_epochs=100)
                metrics = evaluate_model(model, test_loader)

                results = {
                    'Model': model_name,
                    'Combination': combo_key,
                    'Best Validation Loss': best_val_loss,
                    'Best Validation Accuracy': best_val_accuracy,
                    'Test Loss': metrics['test_loss'],
                    'Test Accuracy': metrics['test_accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1 Score': metrics['f1_score'],
                    'AUC': metrics['auc'],
                    'Balanced Accuracy': metrics['balanced_accuracy'],
                    'Specificity': metrics['specificity']
                }

                write_results_to_excel(excel_file_path, results, f'v{iteration + 1}')
                print(f"Finished iteration {iteration + 1} for {model_name} with {combo_key}")

            calculate_and_save_averages(excel_file_path, model_name, combo_key)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
