import yaml
import torch
import numpy as np
import torch.nn.functional as F
import os
from collections import Counter
from torch.utils.data import DataLoader
from vtbench.CNN_train import train_model, create_dataloaders
from vtbench.data_utils import read_ucr, normalize_data, apply_smote, to_torch_tensors
from vtbench.models.SimpleCNN import Simple2DCNN
from vtbench.models.DeepCNN import Deep2DCNN
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier  # XGBoost meta-learner

# Define model configurations
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

def generate_meta_features(models, loader, device):
    """Generate meta-features for the meta-learner using base model predictions."""
    meta_features = []
    true_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            true_labels.extend(labels.cpu().numpy())

            batch_predictions = []
            for model in models:
                model.eval()
                outputs = model(images)
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                batch_predictions.append(probs)

            meta_features.append(np.hstack(batch_predictions))

    meta_features = np.vstack(meta_features)
    true_labels = np.array(true_labels)

    return meta_features, true_labels

def train_meta_learner(meta_features, true_labels):
    """Train the XGBoost meta-learner."""
    meta_learner = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    meta_learner.fit(meta_features, true_labels)
    return meta_learner

def stacking_predict(meta_learner, meta_features):
    """Generate predictions using the meta-learner."""
    final_predictions = meta_learner.predict(meta_features)
    return final_predictions

def evaluate_stacking(models, dataloaders, meta_learner, device):
    """Evaluate the stacking ensemble with XGBoost and compute all metrics including balanced accuracy."""
    test_loader = dataloaders[list(dataloaders.keys())[0]][1]

    # Generate meta-features for the test set
    meta_features, true_labels = generate_meta_features(models, test_loader, device)

    # Predict class probabilities and labels using the meta-learner
    final_probs = meta_learner.predict_proba(meta_features)  # Get probabilities for AUC
    final_predictions = np.argmax(final_probs, axis=1)  # Get predicted class labels

    # Confusion Matrix
    conf_matrix = confusion_matrix(true_labels, final_predictions)
    print(f"Confusion Matrix:\n{conf_matrix}")

    # Calculate test accuracy
    test_accuracy = 100 * np.mean(final_predictions == true_labels)
    print(f"Test Accuracy: {test_accuracy:.2f}%")

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
    balanced_acc = (avg_recall + avg_specificity) / 2

    # Calculate precision, recall, F1 score
    precision = precision_score(true_labels, final_predictions, average='weighted', zero_division=0)
    recall = recall_score(true_labels, final_predictions, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, final_predictions, average='weighted')

    # Calculate AUC using class probabilities
    try:
        auc = roc_auc_score(true_labels, final_probs, multi_class='ovr')
    except ValueError as e:
        print(f"AUC Error: {e}")
        auc = 0.0

    # Print all metrics
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'AUC: {auc:.2f}')
    print(f'Avg Specificity: {avg_specificity:.2f}')
    print(f'Balanced Accuracy: {balanced_acc * 100:.2f}%')

    return test_accuracy, precision, recall, f1, auc, balanced_acc


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    global dataloaders
    train_file = 'data/ECG5000/ECG5000_TRAIN.ts'
    test_file = 'data/ECG5000/ECG5000_TEST.ts'

    x_train, y_train = read_ucr(train_file)
    x_test, y_test = read_ucr(test_file)

    X_train, X_test = normalize_data(x_train, x_test)
    apply_smote_flag = any(model['apply_smote'] for model in config['models'])

    if apply_smote_flag:
        smote_params = config['models'][0]['smote_params']
        X_train_resampled, y_train_resampled = apply_smote(X_train, y_train, smote_params)
        X_train, y_train = X_train_resampled, y_train_resampled

    X_train, y_train, X_test, y_test = to_torch_tensors(X_train, y_train, X_test, y_test)
    dataloaders = create_dataloaders(X_train, y_train, X_test, y_test)

    models = [initialize_model(model_config) for model_config in config['models']]

    # Generate meta-features for the validation set
    val_loader = dataloaders[list(dataloaders.keys())[0]][0]
    meta_features, true_labels = generate_meta_features(models, val_loader, device)

    # Train the XGBoost meta-learner
    meta_learner = train_meta_learner(meta_features, true_labels)

    # Evaluate the stacking ensemble
    evaluate_stacking(models, dataloaders, meta_learner, device)

if __name__ == "__main__":
    main()
