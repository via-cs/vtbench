import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import itertools
from collections import Counter
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
from vtbench.mcCNN_utils import TimeSeriesImageDatasetMC
from vtbench.data_utils import apply_smote
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix, precision_recall_curve, balanced_accuracy_score
from imblearn.metrics import specificity_score
from sklearn.model_selection import StratifiedShuffleSplit
torch.backends.cudnn.enabled = False
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

def stratified_split(test_dataset, y_test, val_size=500):
    y_test = np.array(y_test)
    unique, counts = np.unique(y_test, return_counts=True)
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
    indices = list(range(len(test_dataset)))
    
    for test_idx, val_idx in sss.split(indices, y_test):
        val_dataset = torch.utils.data.Subset(test_dataset, val_idx)  
        test_dataset = torch.utils.data.Subset(test_dataset, test_idx) 
        
        val_labels = [y_test[i] for i in val_idx]
        test_labels = [y_test[i] for i in test_idx]
        
        val_unique, val_counts = np.unique(val_labels, return_counts=True)
        test_unique, test_counts = np.unique(test_labels, return_counts=True)
        
    return val_dataset, test_dataset

def shuffle_indices(length, seed=42):
    np.random.seed(seed)
    indices = np.arange(length)
    np.random.shuffle(indices)
    return indices

def print_class_distribution(y, dataset_name=""):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    print(f"\nClass Distribution for {dataset_name}:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"Class {cls}: {count} samples")


def create_dataloaders(X_train, y_train, X_test, y_test, config, seed = 42):
    torch.manual_seed(seed)
    np.random.seed(seed) 
    dataloaders = {}

    total_samples = {"train": 0, "val": 0, "test": 0}

    shuffled_train_indices = shuffle_indices(len(y_train), seed)
    shuffled_test_indices = shuffle_indices(len(y_test), seed)

    X_train, y_train = X_train[shuffled_train_indices], y_train[shuffled_train_indices]
    X_test, y_test = X_test[shuffled_test_indices], y_test[shuffled_test_indices]

    for model_config in config['models']:
        chart_type = model_config['chart_type'] 
        color_mode = model_config.get('color_mode', 'color')  
        label_mode = model_config.get('label_mode', 'with_label')  
        scatter_mode = model_config.get('scatter_mode', '')  
        bar_mode = model_config.get('bar_mode', '')  

        if chart_type == 'bar':
            combo_key = f"{chart_type}_{bar_mode}_{color_mode}_{label_mode}"
        elif chart_type == 'scatter':
            combo_key = f"{chart_type}_{scatter_mode}_{color_mode}_{label_mode}"
        else:
            combo_key = f"{chart_type}_{color_mode}_{label_mode}"

        print(f"Creating DataLoader for combo key: {combo_key}")

        transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),  
        ])

        if config['apply_smote']:
            print(f"Applying SMOTE for {combo_key}...")
            desired_samples = config['desired_samples_per_class']
            X_resampled, y_resampled = apply_smote(X_train, y_train, desired_samples)
        else:
            print(f"Running without SMOTE for {combo_key}...")
            X_resampled, y_resampled = X_train, y_train

        

        train_dataset = TimeSeriesImageDatasetMC(
            X_resampled, y_resampled, split='train', chart_type=chart_type, transform=transform,
            color_mode=color_mode, label_mode=label_mode,
            scatter_mode=scatter_mode, bar_mode=bar_mode
        )
        full_test_dataset = TimeSeriesImageDatasetMC(
            X_test, y_test, split='test', chart_type=chart_type, transform=transform,
            color_mode=color_mode, label_mode=label_mode,
            scatter_mode=scatter_mode, bar_mode=bar_mode
        )
        
        val_dataset, test_dataset = stratified_split(full_test_dataset, y_test, val_size=500)
        # print_class_distribution([sample[1] for sample in val_dataset], "Validation Dataset")
        # print_class_distribution([sample[1] for sample in test_dataset], "Test Dataset")

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        dataloaders[combo_key] = {'train': train_loader, 'val': val_loader, 'test': test_loader}
        total_samples["train"] += len(train_dataset)
        total_samples["val"] += len(val_dataset)
        total_samples["test"] += len(test_dataset)

    return dataloaders

def load_config(config_path="config.yaml"):
    """Load the configuration from the YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train_joint_cnn(
    model,
    dataloaders,
    combo_keys,
    num_epochs=500,
    patience=15,
    optimizer=None,
    scheduler=None,
):
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_val_loss = float('inf')
    best_val_accuracy = 0
    best_model_state = None
    patience_counter = 0
    
    total_train_samples = sum(len(dataloaders[key]['train'].dataset) for key in combo_keys)
    total_val_samples = sum(len(dataloaders[key]['val'].dataset) for key in combo_keys)
    print(f"\nTotal samples:")
    print(f"Training: {total_train_samples}")
    print(f"Validation: {total_val_samples}")

    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batches in enumerate(zip(*(dataloaders[key]['train'] for key in combo_keys))):
            inputs = []
            labels = None
            
            for i, batch in enumerate(batches):
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                inputs.append(x)
                if labels is None:
                    labels = y
            
            optimizer.zero_grad()
            outputs, attention_weights = model(*inputs)  # Unpack tuple here
            loss = criterion(outputs, labels)  # Use only the outputs
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / total
        epoch_accuracy = 100 * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, batches in enumerate(zip(*(dataloaders[key]['val'] for key in combo_keys))):
                inputs = []
                labels = None
                
                for i, batch in enumerate(batches):
                    x, y = batch
                    x = x.to(device)
                    y = y.to(device)
                    inputs.append(x)
                    if labels is None:
                        labels = y
                
                outputs, _ = model(*inputs)  # Unpack tuple here too
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / val_total
        val_accuracy = 100 * val_correct / val_total
        
        if scheduler is not None:
            scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"Training completed. Best validation accuracy: {best_val_accuracy:.2f}%")
    return model, None  # Return None as history for now


def test_model(model, test_dataloaders, criterion, combo_keys, device):
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds, all_probs = [], [], []
   
    total_test_samples = len(next(iter(test_dataloaders.values()))['test'].dataset)
    print(f"Expected Total Test Samples: {total_test_samples}")
    
    with torch.no_grad():
        for batch_idx, batches in enumerate(zip(*(test_dataloaders[key]['test'] for key in combo_keys))):
            inputs, batch_labels = [], None
            
            for i, batch in enumerate(batches):
                x, labels = batch  
                x = x.to(device)
                labels = labels.to(device)
                inputs.append(x)
                if batch_labels is None:
                    batch_labels = labels  
                elif not torch.equal(batch_labels, labels):
                    raise ValueError(f"Label mismatch in batch {batch_idx} for branch {combo_keys[i]}")
            
            # Unpack the model output tuple
            outputs, attention_weights = model(*inputs)
            
            loss = criterion(outputs, batch_labels)
            test_loss += loss.item() * batch_labels.size(0)
           
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
           
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.extend(probs)
            print(f"Batch {batch_idx + 1} processed")

    
    print(f"Total Processed Test Samples: {total}")
    if total != total_test_samples:
        raise ValueError("Mismatch in processed and expected test samples!")

   
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Total samples in Confusion Matrix: {conf_matrix.sum()}")

   
    test_loss /= total
    test_accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    auc = roc_auc_score(all_labels, np.array(all_probs), multi_class='ovr')

    
    recall_per_class, specificity_per_class = [], []
    for i in range(len(conf_matrix)):
        recall_i = conf_matrix[i, i] / conf_matrix[i, :].sum() if conf_matrix[i, :].sum() > 0 else 0
        tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
        fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
        specificity_i = tn / (tn + fp) if (tn + fp) > 0 else 0

        recall_per_class.append(recall_i)
        specificity_per_class.append(specificity_i)

    avg_recall = np.mean(recall_per_class)
    avg_specificity = np.mean(specificity_per_class)
    balanced_acc = (avg_recall + avg_specificity) / 2

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    print(f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}')
    print(f'AUC: {auc:.2f}, Specificity: {avg_specificity:.2f}')
    print(f'Balanced Accuracy: {balanced_acc * 100:.2f}%')

    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'specificity': avg_specificity,
        'balanced_accuracy': balanced_acc,
        'confusion_matrix': conf_matrix
    }








