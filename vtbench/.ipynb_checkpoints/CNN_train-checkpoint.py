import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
from vtbench.mcCNN_utils import TimeSeriesImageDatasetMC
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix, balanced_accuracy_score

torch.backends.cudnn.enabled = False

def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    chart_types = ['area', 'line', 'bar', 'scatter'] 
    color_modes = ['color', 'monochrome']
    label_modes = ['with_label', 'without_label']
    scatter_modes = ['plain', 'join']  # Specific to scatter charts
    bar_modes = ['fill', 'border'] 
   
    dataloaders = {}

    for chart_type in chart_types:
        for color_mode in color_modes:
            for label_mode in label_modes:
                if chart_type == 'scatter':
                    for scatter_mode in scatter_modes:
                        print(f"\nRunning model with chart_type: {chart_type}, scatter_mode: {scatter_mode}, color_mode: {color_mode}, label_mode: {label_mode}")
    
                        train_dataset = TimeSeriesImageDatasetMC(X_train.numpy(), y_train.numpy(), split='train', transform=transform, 
                                                             chart_type=chart_type, color_mode=color_mode, label_mode=label_mode, scatter_mode=scatter_mode)
                        test_dataset = TimeSeriesImageDatasetMC(X_test.numpy(), y_test.numpy(), split='test', transform=transform, 
                                                            chart_type=chart_type, color_mode=color_mode, label_mode=label_mode, scatter_mode=scatter_mode)

                        # Split the test data into validation and test sets
                        val_size = 500
                        test_size = len(test_dataset) - val_size
                        val_dataset, test_dataset = random_split(test_dataset, [val_size, test_size])
    
                        # Create DataLoaders
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
                        # Store the DataLoaders in the dictionary
                        combo_key = f"{chart_type}_{scatter_mode}_{color_mode}_{label_mode}"
                        dataloaders[combo_key] = (train_loader, val_loader, test_loader)

                elif chart_type == 'bar':
                    for bar_mode in bar_modes:
                        print(f"\nRunning model with chart_type: {chart_type}, bar_mode: {bar_mode}, color_mode: {color_mode}, label_mode: {label_mode}")
    
                        # Create datasets with the current combination
                        train_dataset = TimeSeriesImageDatasetMC(X_train.numpy(), y_train.numpy(), split='train', transform=transform, 
                                                                 chart_type=chart_type, color_mode=color_mode, label_mode=label_mode, bar_mode=bar_mode)
                        test_dataset = TimeSeriesImageDatasetMC(X_test.numpy(), y_test.numpy(), split='test', transform=transform, 
                                                                chart_type=chart_type, color_mode=color_mode, label_mode=label_mode, bar_mode=bar_mode)
    
                        # Split the test data into validation and test sets
                        val_size = int(0.5 * len(test_dataset))
                        test_size = len(test_dataset) - val_size
                        val_dataset, test_dataset = random_split(test_dataset, [val_size, test_size])
    
                        # Create DataLoaders
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
                        # Store the DataLoaders in the dictionary
                        combo_key = f"{chart_type}_{bar_mode}_{color_mode}_{label_mode}"
                        dataloaders[combo_key] = (train_loader, val_loader, test_loader)
            
                else:
                    print(f"\nRunning model with chart_type: {chart_type}, color_mode: {color_mode}, label_mode: {label_mode}")
    
                    # Create datasets with the current combination
                    train_dataset = TimeSeriesImageDatasetMC(X_train.numpy(), y_train.numpy(), split='train', transform=transform, 
                                                             chart_type=chart_type, color_mode=color_mode, label_mode=label_mode)
                    test_dataset = TimeSeriesImageDatasetMC(X_test.numpy(), y_test.numpy(), split='test', transform=transform, 
                                                            chart_type=chart_type, color_mode=color_mode, label_mode=label_mode)
    
                    # Split the test data into validation and test sets
                    val_size = int(0.5 * len(test_dataset))
                    test_size = len(test_dataset) - val_size
                    val_dataset, test_dataset = random_split(test_dataset, [val_size, test_size])
    
                    # Create DataLoaders
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
                    # Store the DataLoaders in the dictionary
                    combo_key = f"{chart_type}_{color_mode}_{label_mode}"
                    dataloaders[combo_key] = (train_loader, val_loader, test_loader)

    return dataloaders

def train_model(model, train_loader, val_loader, num_epochs, patience=10):
    criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

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

    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    auc = roc_auc_score(y_true, y_probs, multi_class='ovr')
    conf_matrix = confusion_matrix(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    specificity = []
    for i in range(len(conf_matrix)):
        tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
        fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
        specificity.append(tn / (tn + fp))

    avg_specificity = sum(specificity) / len(specificity)

    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'AUC: {auc:.2f}')
    print(f'Balanced Accuracy: {balanced_acc:.2f}')
    print(f'Average Specificity: {avg_specificity:.2f}')
    print('Confusion Matrix:')
    print(conf_matrix)

    return test_loss, test_accuracy


def plot_class_distribution(y_train, nb_classes):
    plt.hist(y_train.numpy(), bins=nb_classes, edgecolor='k')
    plt.title('Class Distribution in Training Set')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.xticks(ticks=np.arange(nb_classes))
    plt.show()