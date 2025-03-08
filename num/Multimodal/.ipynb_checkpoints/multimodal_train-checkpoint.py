import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
from collections import Counter
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
from multimodal_utils import TimeSeriesImageDatasetMC, augmentation_transforms
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix, precision_recall_curve, balanced_accuracy_score
from imblearn.metrics import specificity_score
from sklearn.model_selection import StratifiedShuffleSplit
import random
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False


from sklearn.model_selection import StratifiedShuffleSplit

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

config = load_config() 

def get_dataset_name_from_path(path):
    """Extract the dataset name from the path."""
    return os.path.basename(os.path.dirname(path))
    

def stratified_split(test_dataset, y_test, val_size=0.2):
    """Perform stratified split with a percentage-based validation size."""
    y_test = np.array(y_test)  
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)

    indices = np.arange(len(test_dataset))
    for train_idx, val_idx in sss.split(indices, y_test):
        val_dataset = torch.utils.data.Subset(test_dataset, val_idx)
        test_dataset = torch.utils.data.Subset(test_dataset, train_idx)

    return val_dataset, test_dataset


def create_dataloaders(X_train, numerical_train, y_train, X_test, numerical_test, y_test, train_file, batch_size=32):
    dataset_name = get_dataset_name_from_path(train_file)
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    multimodal_config = config.get("multimodal", {})

    chart_type = multimodal_config.get("chart_type", "scatter")
    color_mode = multimodal_config.get("color_mode", "monochrome")
    label_mode = multimodal_config.get("label_mode", "with_label")
    scatter_mode = multimodal_config.get("scatter_mode", "plain")
    bar_mode = multimodal_config.get("bar_mode", "fill")

    dataloaders = {}
    combo_key = f"{chart_type}_{color_mode}_{label_mode}"
    
    if chart_type == "scatter":
        combo_key += f"_{scatter_mode}"
    elif chart_type == "bar":
        combo_key += f"_{bar_mode}"

    print(f"Creating dataloaders for: {combo_key}")
    
    try:
        train_dataset = TimeSeriesImageDatasetMC(
            dataset_name=dataset_name,
            time_series_data=X_train,
            numerical_data=numerical_train,
            labels=y_train,
            split='train',
            transform=transform,
            chart_type=chart_type,
            color_mode=color_mode,
            label_mode=label_mode,
            scatter_mode=scatter_mode if chart_type == "scatter" else None,
            bar_mode=bar_mode if chart_type == "bar" else None
        )
        test_dataset = TimeSeriesImageDatasetMC(
            dataset_name=dataset_name,
            time_series_data=X_test,
            numerical_data=numerical_test,
            labels=y_test,
            split='test',
            transform=transform,
            chart_type=chart_type,
            color_mode=color_mode,
            label_mode=label_mode,
            scatter_mode=scatter_mode if chart_type == "scatter" else None,
            bar_mode=bar_mode if chart_type == "bar" else None
        )

        val_size = int(0.2 * len(test_dataset)) 
        val_size = max(1, val_size) 
        
        val_dataset, test_dataset = stratified_split(test_dataset, y_test, val_size=val_size)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        dataloaders[combo_key] = (train_loader, val_loader, test_loader)

    except Exception as e:
        print(f"Error occurred while creating dataloaders for combination: {combo_key}")
        print(f"Error: {e}")

    print(f"Available keys in dataloaders: {dataloaders.keys()}")
    return dataloaders

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(model, train_loader, val_loader, num_epochs, patience=10, optimizer=None, scheduler=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.01)
    
    if scheduler is None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = 0
    trigger_times = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, numerical_data, labels in train_loader:
            images, numerical_data, labels = images.to(device), numerical_data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, numerical_data)  
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.2f}%')

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.inference_mode():
            for images, numerical_data, labels in val_loader:
                images, numerical_data, labels = images.to(device), numerical_data.to(device), labels.to(device)
                outputs = model(images, numerical_data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        scheduler.step(val_loss)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    print(f'Best Validation Accuracy: {best_val_accuracy:.2f}%')
    return model, val_loss, best_val_accuracy

def evaluate_model(model, test_loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    y_true = []
    y_pred = []
    y_probs = []

    with torch.inference_mode():
        for images, numerical_data, labels in test_loader:
            images, numerical_data, labels = images.to(device), numerical_data.to(device), labels.to(device)
            outputs = model(images, numerical_data)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy()[:, 1])

    test_loss /= len(test_loader)
    test_accuracy = 100 * correct / total
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)
    specificity = specificity_score(y_true, y_pred)

    conf_matrix = confusion_matrix(y_true, y_pred)

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}%')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'AUC: {auc:.4f}')
    print(f'Specificity: {specificity:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)

    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'specificity': specificity,
        'confusion_matrix': conf_matrix
    }