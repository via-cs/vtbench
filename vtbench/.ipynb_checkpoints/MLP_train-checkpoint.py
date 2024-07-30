import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from vtbench.data_utils import read_ucr, normalize_data, apply_smote, to_torch_tensors
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix, balanced_accuracy_score
from vtbench.models.MLP import MLP

def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=32):
    # Create full datasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # Split the test data into validation and test sets
    val_size = int(0.5 * len(test_dataset))
    test_size = len(test_dataset) - val_size
    val_dataset, test_dataset = random_split(test_dataset, [val_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, num_epochs, patience=10):
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
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
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
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
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
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
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

    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'AUC: {auc:.2f}')
    print(f'Balanced Accuracy: {balanced_acc:.2f}')
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