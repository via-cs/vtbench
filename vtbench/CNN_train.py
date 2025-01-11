import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import matplotlib.pyplot as plt
from vtbench.mcCNN_utils import TimeSeriesImageDatasetMC
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import StratifiedShuffleSplit

torch.backends.cudnn.enabled = False

from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset

def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    print(f"Original test dataset size: {len(X_test)}")

    # First, split the original test set into a 500-sample validation set and the rest for testing
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=500, random_state=42)
    val_indices, test_indices = next(stratified_split.split(X_test, y_test))

    # Create validation and test datasets from the original test set
    val_dataset = Subset(
        TimeSeriesImageDatasetMC(X_test, y_test, split='val', transform=transform),
        val_indices
    )
    test_dataset = Subset(
        TimeSeriesImageDatasetMC(X_test, y_test, split='test', transform=transform),
        test_indices
    )

    chart_types = ['area', 'line', 'bar', 'scatter']
    color_modes = ['color', 'monochrome']
    label_modes = ['with_label', 'without_label']
    scatter_modes = ['plain', 'join']
    bar_modes = ['fill', 'border']

    dataloaders = {}

    for chart_type in chart_types:
        for color_mode in color_modes:
            for label_mode in label_modes:
                if chart_type == 'scatter':
                    for scatter_mode in scatter_modes:
                        train_dataset = TimeSeriesImageDatasetMC(
                            X_train, y_train, split='train', transform=transform,
                            chart_type=chart_type, color_mode=color_mode,
                            label_mode=label_mode, scatter_mode=scatter_mode
                        )

                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
                        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

                        combo_key = f"{chart_type}_{scatter_mode}_{color_mode}_{label_mode}"
                        dataloaders[combo_key] = (train_loader, val_loader, test_loader)

                elif chart_type == 'bar':
                    for bar_mode in bar_modes:
                        train_dataset = TimeSeriesImageDatasetMC(
                            X_train, y_train, split='train', transform=transform,
                            chart_type=chart_type, color_mode=color_mode,
                            label_mode=label_mode, bar_mode=bar_mode
                        )

                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
                        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

                        combo_key = f"{chart_type}_{bar_mode}_{color_mode}_{label_mode}"
                        dataloaders[combo_key] = (train_loader, val_loader, test_loader)

                else:
                    train_dataset = TimeSeriesImageDatasetMC(
                        X_train, y_train, split='train', transform=transform,
                        chart_type=chart_type, color_mode=color_mode, label_mode=label_mode
                    )

                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

                    combo_key = f"{chart_type}_{color_mode}_{label_mode}"
                    dataloaders[combo_key] = (train_loader, val_loader, test_loader)

    return dataloaders



def train_model(model, train_loader, val_loader, num_epochs, patience=10, optimizer=None, scheduler=None):

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.01)
    
    if scheduler is None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
 
    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = 0
    trigger_times = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
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
            for images, labels in val_loader:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        scheduler.step(val_loss)

        final_val_loss = val_loss

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    print(f'Best Validation Accuracy: {best_val_accuracy:.2f}%')
    print(f'Validation Loss: {final_val_loss:.4f}')  

    return final_val_loss, best_val_accuracy

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

    num_test_samples = sum(len(batch[0]) for batch in test_loader)

    with torch.inference_mode():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    # Ensuring the sizes match
    assert len(y_true) == len(y_pred), f"Mismatch: y_true({len(y_true)}) != y_pred({len(y_pred)})"

    # Calculate test metrics
    test_loss /= len(test_loader)
    test_accuracy = 100 * correct / total
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    # Calculate confusion matrix and per-class metrics
    conf_matrix = confusion_matrix(y_true, y_pred)
    recall_per_class = []
    specificity_per_class = []

    for i in range(len(conf_matrix)):
        recall_i = (
            conf_matrix[i, i] / conf_matrix[i, :].sum()
            if conf_matrix[i, :].sum() > 0 else 0
        )
        tn = conf_matrix.sum() - (
            conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i]
        )
        fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
        specificity_i = tn / (tn + fp) if (tn + fp) > 0 else 0

        recall_per_class.append(recall_i)
        specificity_per_class.append(specificity_i)

    # Average metrics
    avg_recall = np.mean(recall_per_class)
    avg_specificity = np.mean(specificity_per_class)
    balanced_acc = (avg_recall + avg_specificity) / 2

    # Final metrics
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted')
    auc = roc_auc_score(y_true, y_probs, multi_class='ovr')

    # Display results
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'AUC: {auc:.2f}')
    print(f'Specificity: {avg_specificity:.2f}')
    print(f'Balanced Accuracy: {balanced_acc * 100:.2f}%')
    print('Confusion Matrix:')
    print(conf_matrix)

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


def plot_precision_recall_curve(y_true, y_probs, model_name, smote_status, iteration, chart_type, color_mode, label_mode):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs[:, 1])

    # Plot the precision-recall curve
    plt.plot(thresholds, precision[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'Precision-Recall vs Threshold\nModel: {model_name}, SMOTE: {smote_status}, Chart: {chart_type}, Color: {color_mode}, Labels: {label_mode}, Iteration: {iteration}')
    plt.legend(loc='best')

    plt.savefig(f'precision_recall_curve_{model_name}_{smote_status}_{chart_type}_{color_mode}_{label_mode}_iter{iteration}.png')
    plt.close()

def plot_class_distribution(y_train, nb_classes):
    plt.hist(y_train.numpy(), bins=nb_classes, edgecolor='k')
    plt.title('Class Distribution in Training Set')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.xticks(ticks=np.arange(nb_classes))
    plt.show()