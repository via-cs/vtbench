import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
from vtbench.mcCNN_utils import TimeSeriesImageDatasetMC
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix, precision_recall_curve

torch.backends.cudnn.enabled = False

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=None, gamma=2, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha  # If alpha is None, the loss will not apply class weighting
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, inputs, targets):
#         BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
#         pt = torch.exp(-BCE_loss)
#         if self.alpha is not None:
#             at = self.alpha.gather(0, targets.data.view(-1))
#             F_loss = at * (1 - pt) ** self.gamma * BCE_loss
#         else:
#             F_loss = (1 - pt) ** self.gamma * BCE_loss

#         if self.reduction == 'mean':
#             return F_loss.mean()
#         elif self.reduction == 'sum':
#             return F_loss.sum()
#         else:
#             return F_loss


def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    chart_types = ['area', 'line', 'bar', 'scatter']
    color_modes = ['color', 'monochrome']
    label_modes = ['with_label', 'without_label']
    scatter_modes = ['plain', 'join']
    bar_modes = ['fill', 'border']

    dataloaders = {}

    # Iterate over chart types, color modes, etc.
    for chart_type in chart_types:
        for color_mode in color_modes:
            for label_mode in label_modes:
                if chart_type == 'scatter':
                    for scatter_mode in scatter_modes:
                        # Create dataset and loaders for scatter mode
                        train_dataset = TimeSeriesImageDatasetMC(
                            X_train, y_train, split='train', transform=transform,
                            chart_type=chart_type, color_mode=color_mode, label_mode=label_mode, scatter_mode=scatter_mode
                        )
                        test_dataset = TimeSeriesImageDatasetMC(
                            X_test, y_test, split='test', transform=transform,
                            chart_type=chart_type, color_mode=color_mode, label_mode=label_mode, scatter_mode=scatter_mode
                        )

                        # Split test data
                        val_size = 500
                        test_size = len(test_dataset) - val_size
                        val_dataset, test_dataset = random_split(test_dataset, [val_size, test_size])

                        # Create DataLoaders
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                        # Store DataLoaders
                        combo_key = f"{chart_type}_{scatter_mode}_{color_mode}_{label_mode}"
                        dataloaders[combo_key] = (train_loader, val_loader, test_loader)


                elif chart_type == 'bar':
                    for bar_mode in bar_modes:
                        # Create dataset and loaders for bar charts
                        train_dataset = TimeSeriesImageDatasetMC(
                            X_train, y_train, split='train', transform=transform,
                            chart_type=chart_type, color_mode=color_mode, label_mode=label_mode, bar_mode=bar_mode
                        )
                        test_dataset = TimeSeriesImageDatasetMC(
                            X_test, y_test, split='test', transform=transform,
                            chart_type=chart_type, color_mode=color_mode, label_mode=label_mode, bar_mode=bar_mode
                        )

                        val_size = int(0.5 * len(test_dataset))
                        test_size = len(test_dataset) - val_size
                        val_dataset, test_dataset = random_split(test_dataset, [val_size, test_size])

                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                        combo_key = f"{chart_type}_{bar_mode}_{color_mode}_{label_mode}"
                        dataloaders[combo_key] = (train_loader, val_loader, test_loader)

    

                else:
                    # General case for other chart types
                    train_dataset = TimeSeriesImageDatasetMC(
                        X_train, y_train, split='train', transform=transform,
                        chart_type=chart_type, color_mode=color_mode, label_mode=label_mode
                    )
                    test_dataset = TimeSeriesImageDatasetMC(
                        X_test, y_test, split='test', transform=transform,
                        chart_type=chart_type, color_mode=color_mode, label_mode=label_mode
                    )

                    val_size = int(0.5 * len(test_dataset))
                    test_size = len(test_dataset) - val_size
                    val_dataset, test_dataset = random_split(test_dataset, [val_size, test_size])

                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                    combo_key = f"{chart_type}_{color_mode}_{label_mode}"
                    dataloaders[combo_key] = (train_loader, val_loader, test_loader)



    return dataloaders


def train_model(model, train_loader, val_loader, num_epochs, patience=10, optimizer=None, scheduler=None):

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.01)
    
    if scheduler is None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # alpha = torch.tensor([0, 0.05, 0.1, 0.5, 1.0]).to(device)  # Adjust according to your class distribution
    # gamma = 2  # Adjust based on results; gamma = 2 is a good starting point
    
    class_weights = torch.tensor([1.0, 1.2, 3.0, 4.5, 5.0]).to(device)  
    criterion = nn.CrossEntropyLoss(weight = class_weights)

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

    test_loss /= len(test_loader)
    test_accuracy = 100 * correct / total
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    precision = precision_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    auc = roc_auc_score(y_true, y_probs, multi_class='ovr')
    conf_matrix = confusion_matrix(y_true, y_pred)

    recall_per_class = []
    specificity_per_class = []

    for i in range(len(conf_matrix)):
        # Recall (Sensitivity) for class i: TP / (TP + FN)
        recall_i = conf_matrix[i, i] / conf_matrix[i, :].sum()
        recall_per_class.append(recall_i)
        
        # Specificity for class i: TN / (TN + FP)
        tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
        fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
        specificity_i = tn / (tn + fp)
        specificity_per_class.append(specificity_i)

    # Calculate average recall and average specificity
    avg_recall = sum(recall_per_class) / len(recall_per_class)
    avg_specificity = sum(specificity_per_class) / len(specificity_per_class)

    # Balanced Accuracy: average of recall and specificity per class
    balanced_acc = (avg_recall + avg_specificity) / 2

    print(f'Precision (Macro): {precision:.2f}')
    print(f'Recall (Macro): {avg_recall:.2f}')
    print(f'F1 Score (Macro): {f1:.2f}')
    print(f'AUC: {auc:.2f}')
    print(f'Balanced Accuracy: {balanced_acc:.2f}')
    print(f'Average Specificity: {avg_specificity:.2f}')
    print('Confusion Matrix:')
    print(conf_matrix)

    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': avg_recall,  
        'f1_score': f1,
        'auc': auc,
        'avg_specificity': avg_specificity,
        'balanced_accuracy': balanced_acc,
        'y_true': y_true
    }

def evaluate_model_cv(model, val_loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    y_true = []
    y_pred = []

    with torch.inference_mode():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
    print(f'Balanced Accuracy: {balanced_acc:.2f}')

    return {
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'balanced_accuracy': balanced_acc
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