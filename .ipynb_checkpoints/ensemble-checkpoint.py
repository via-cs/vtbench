import yaml
import torch
import torch.nn.functional as F
import os
from collections import Counter
import numpy as np
import json
from collections import defaultdict
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
)
from vtbench.CNN_train import train_model, create_dataloaders
from vtbench.data_utils import read_ucr, normalize_data, apply_smote, to_torch_tensors
from vtbench.models.SimpleCNN import Simple2DCNN
from vtbench.models.DeepCNN import Deep2DCNN


model_configurations = {
    'Simple2DCNN': Simple2DCNN,
    'Deep2DCNN': Deep2DCNN
}

def load_config(config_file="ensembleconfig.yaml"):
    """Load the YAML configuration file."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def initialize_model(model_config):
    """Initialize and load a model based on the configuration."""
    chart_key = model_config['chart_type']
    model_type = model_config['model_type']
    input_channels = model_config['input_channels']
    num_classes = model_config['num_classes']

   
    smote_suffix = "_smote" if model_config.get('apply_smote', False) else ""
    model_path = f"trained_models/{chart_key}{smote_suffix}.pt"

   
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model = torch.load(model_path)
    else:
        print(f"{model_path} not found. Training a new model...")
        model = get_model(model_type, input_channels, num_classes)
        model = train_and_save_model(model, model_config, model_path)

    return model

def get_model(model_type, input_channels, num_classes):
    """Get a model instance based on its type."""
    if model_type in model_configurations:
        return model_configurations[model_type](input_channels, num_classes)
    else:
        raise ValueError(f"Model type '{model_type}' is not supported.")

def train_and_save_model(model, model_config, model_path):
    """Train and save a model to disk."""
    chart_key = model_config['chart_type']
    train_loader, val_loader, _ = dataloaders[chart_key]

    print(f"Training model for chart key: {chart_key}...")
    _, best_accuracy = train_model(model, train_loader, val_loader, num_epochs=20)

    print(f"Saving model with {best_accuracy:.2f}% accuracy to {model_path}...")
    torch.save(model, model_path)

    return model

def majority_vote(predictions):
    """Perform majority voting on predictions."""
    voted_predictions = [Counter(pred).most_common(1)[0][0] for pred in zip(*predictions)]
    return voted_predictions

def evaluate_ensemble(models, model_configs, dataloaders, device):
    """Evaluate the ensemble using majority voting and compute metrics, saving only ensemble results for visualization."""
    ensemble_results_dict = {"Ensemble Majority Voting": {"true_labels": [], "predicted_labels": []}}
    all_probs = []
    all_predictions = [[] for _ in models]
    true_labels = []
    total_loss = 0.0
    total_samples = 0

    test_loader = dataloaders[list(dataloaders.keys())[0]][1]
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in test_loader:
            batch_size = labels.size(0)
            total_samples += batch_size

            images, labels = images.to(device), labels.to(device)
            batch_true_labels = labels.cpu().numpy().tolist()  # Convert to Python list of int
            true_labels.extend(batch_true_labels)

            batch_probs = []

            # Collect probabilities and predictions from each model
            for i, model in enumerate(models):
                model.eval()
                outputs = model(images)
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                batch_probs.append(probs)

                # Predicted classes for this model
                _, predicted = torch.max(outputs, 1)
                batch_predicted_labels = predicted.cpu().numpy().tolist()  # Convert to Python list of int
                all_predictions[i].extend(batch_predicted_labels)

                # Calculate loss
                loss = criterion(outputs, labels)
                total_loss += loss.item() * batch_size

            # Average the probabilities across models for each sample
            avg_probs = np.mean(batch_probs, axis=0)
            all_probs.extend(avg_probs)

    # Perform majority voting for final ensemble prediction
    final_predictions = majority_vote(all_predictions)

    # Convert true and predicted labels to lists of native Python ints
    ensemble_results_dict["Ensemble Majority Voting"]["true_labels"] = [int(label) for label in true_labels]
    ensemble_results_dict["Ensemble Majority Voting"]["predicted_labels"] = [int(pred) for pred in final_predictions]

    # Save the ensemble results dictionary to a JSON file for visualization
    with open('ensemble_results_dict.json', 'w') as f:
        json.dump(ensemble_results_dict, f)

    # Calculate accuracy and print other metrics as needed
    test_accuracy = 100 * np.mean(np.array(final_predictions) == np.array(true_labels))
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    avg_test_loss = total_loss / total_samples
    conf_matrix = confusion_matrix(true_labels, final_predictions)
    print(f"Confusion Matrix:\n{conf_matrix}")

    # Calculate recall and specificity for each class
    recall_per_class = []
    specificity_per_class = []

    for i in range(len(conf_matrix)):
        tp = conf_matrix[i, i]
        fn = conf_matrix[i, :].sum() - tp
        fp = conf_matrix[:, i].sum() - tp
        tn = conf_matrix.sum() - (tp + fn + fp)

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        recall_per_class.append(recall)
        specificity_per_class.append(specificity)

    # Calculate average recall, specificity, and balanced accuracy
    avg_recall = np.mean(recall_per_class)
    avg_specificity = np.mean(specificity_per_class)
    balanced_accuracy = avg_recall * 100  # Convert to percentage
    

    # Calculate precision, recall, F1 score
    precision = precision_score(true_labels, final_predictions, average="weighted", zero_division=0)
    recall = recall_score(true_labels, final_predictions, average="weighted", zero_division=0)
    f1 = f1_score(true_labels, final_predictions, average="weighted")

    # Calculate AUC using the averaged probabilities
    try:
        auc = roc_auc_score(true_labels, np.array(all_probs), multi_class='ovr')
    except ValueError as e:
        print(f"AUC Error: {e}")
        auc = 0.0

    # Print additional metrics
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'AUC: {auc:.2f}')
    print(f"Balanced Accuracy: {balanced_accuracy:.2f}%")

    return avg_test_loss, test_accuracy, balanced_accuracy, precision, recall, f1, auc

def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_file = 'data/ECG5000/ECG5000_TRAIN.ts'
    test_file = 'data/ECG5000/ECG5000_TEST.ts'
    x_train, y_train = read_ucr(train_file)
    x_test, y_test = read_ucr(test_file)

    X_train, X_test = normalize_data(x_train, x_test)

    apply_smote_flag = any(model['apply_smote'] for model in config['models'])
    if apply_smote_flag:
        smote_params = config['models'][0].get('smote_params', {})
        X_train, y_train = apply_smote(X_train, y_train, smote_params)

    X_train, y_train, X_test, y_test = to_torch_tensors(X_train, y_train, X_test, y_test)
    dataloaders = create_dataloaders(X_train, y_train, X_test, y_test)

    models = [initialize_model(model_config) for model_config in config['models']]

    evaluate_ensemble(models, config['models'], dataloaders, device)

if __name__ == "__main__":
    main()

