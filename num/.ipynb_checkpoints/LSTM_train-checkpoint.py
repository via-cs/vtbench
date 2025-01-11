import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, 
    roc_auc_score, roc_curve
)


def create_dataloader(X, y, batch_size=32):
    
    if len(X.shape) == 2:  
        X = X.unsqueeze(-1)  

    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def stratified_split(X, y, test_size=0.2):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
    for train_index, val_index in sss.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
    return X_train, X_val, y_train, y_val

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)

       
        loss = criterion(outputs, targets)
        total_loss += loss.item()

        
        predictions = (outputs > 0.5).float()
        correct += (predictions == targets).sum().item()
        total += targets.size(0)

        
        loss.backward()
        optimizer.step()

   
    avg_loss = total_loss / len(train_loader)
    accuracy = (correct / total) * 100 

    print(f"Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy



def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device).float().unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            
            predictions = (outputs > 0.5).float()
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

   
    avg_loss = total_loss / len(val_loader)
    accuracy = (correct / total) * 100  

    print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy



def test_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_probs = []  
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

          
            probs = torch.sigmoid(outputs).cpu().numpy()  
            all_probs.extend(probs)

            
            predictions = (probs > 0.5).astype(int)
            all_preds.extend(predictions)
            all_targets.extend(targets.numpy())

  
    conf_matrix = confusion_matrix(all_targets, all_preds)
    tn, fp, fn, tp = conf_matrix.ravel()
    
    recalls = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    balanced_accuracy = recalls.mean() * 100

    accuracy = accuracy_score(all_targets, all_preds) * 100 
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    auc = roc_auc_score(all_targets, all_probs)

    print(f"\nTest Metrics:")
    print(f"Accuracy: {accuracy:.2f}%, Precision: {precision:.4f}, "
          f"Recall (Sensitivity): {recall:.4f}, Specificity: {specificity:.4f}, "
          f"F1-Score: {f1:.4f}, AUC: {auc:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.2f}%")
    print(f"Confusion Matrix:\n{conf_matrix}")

