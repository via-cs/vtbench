import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import json
from vtbench.data_utils import read_ucr, normalize_data, apply_smote, to_torch_tensors
from vtbench.models.SimpleCNN import Simple2DCNN
from vtbench.models.DeepCNN import Deep2DCNN
from vtbench.CNN_train import create_dataloaders, train_model, evaluate_model
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix, precision_recall_curve

def train_and_evaluate_best_model(combo_key, apply_smote_flag, model_type, X_train, y_train, X_test, y_test, best_models_results_dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    if apply_smote_flag:
        desired_samples_per_class = {  
            1: 200,
            2: 200,
            3: 250,
            4: 300
        }
        X_train_resampled, y_train_resampled = apply_smote(X_train, y_train, desired_samples_per_class)
        X_train, y_train, X_test, y_test = to_torch_tensors(X_train_resampled, y_train_resampled, X_test, y_test)
    else:
        X_train, y_train, X_test, y_test = to_torch_tensors(X_train, y_train, X_test, y_test)

    train_loader, val_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test)[combo_key]

    model = model_type(3, len(torch.unique(y_train))).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 100
    best_val_loss, best_val_accuracy = train_model(model, train_loader, val_loader, num_epochs, patience=10, optimizer=optimizer)

    model.eval()
    y_true, y_pred, y_probs = [], [], []  

    with torch.inference_mode():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1) 
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(probs.cpu().numpy()) 

    y_true = list(map(int, y_true))
    y_pred = list(map(int, y_pred))


    best_models_results_dict[combo_key] = {
        'true_labels': y_true,
        'predicted_labels': y_pred
    }


    precision = precision_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    auc = roc_auc_score(y_true, y_probs, multi_class='ovr')  
    conf_matrix = confusion_matrix(y_true, y_pred)

    recall_per_class = []
    specificity_per_class = []

    for i in range(len(conf_matrix)):
        recall_i = conf_matrix[i, i] / conf_matrix[i, :].sum() if conf_matrix[i, :].sum() > 0 else 0
        recall_per_class.append(recall_i)
        
        tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
        fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
        specificity_i = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_per_class.append(specificity_i)

    avg_recall = np.mean(recall_per_class)
    avg_specificity = np.mean(specificity_per_class)

    balanced_acc = (avg_recall + avg_specificity) / 2 * 100

    print(f'Precision (Macro): {precision:.2f}')
    print(f'Recall (Macro): {avg_recall:.2f}')
    print(f'F1 Score (Macro): {f1:.2f}')
    print(f'AUC: {auc:.2f}')
    print(f'Balanced Accuracy: {balanced_acc:.2f}%')
    print(f'Average Specificity: {avg_specificity:.2f}')
    print('Confusion Matrix:')
    print(conf_matrix)

    return model
    
if __name__ == "__main__":
    best_combinations = [
        {"combo_key": "scatter_plain_monochrome_with_label", "apply_smote": False, "model_type": Deep2DCNN},
        {"combo_key": "bar_border_color_with_label", "apply_smote": True, "model_type": Deep2DCNN},
        {"combo_key": "area_monochrome_with_label", "apply_smote": True, "model_type": Deep2DCNN},
        {"combo_key": "line_color_without_label", "apply_smote": False, "model_type": Deep2DCNN}
    ]
    
    best_models_results_dict = {}
    
    train_file = 'data/ECG5000/ECG5000_TRAIN.ts'
    test_file = 'data/ECG5000/ECG5000_TEST.ts'
    
    x_train, y_train = read_ucr(train_file)
    x_test, y_test = read_ucr(test_file)
    
    x_train, x_test = normalize_data(x_train, x_test)
    
    
    unique_labels = np.unique(y_train)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y_train = np.array([label_map[label] for label in y_train])
    y_test = np.array([label_map[label] for label in y_test])
    
    
    for best_combo in best_combinations:
        combo_key = best_combo["combo_key"]
        apply_smote_flag = best_combo["apply_smote"]
        model_type = best_combo["model_type"]
    
        train_and_evaluate_best_model(combo_key, apply_smote_flag, model_type, x_train, y_train, x_test, y_test, best_models_results_dict)
    
    with open('best_models_results_dict.json', 'w') as f:
        json.dump(best_models_results_dict, f)
    
    print("Results saved to 'best_models_results_dict.json'")