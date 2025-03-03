import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

def prepare_dataloaders(x_train, y_train, x_test, y_test, batch_size):
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def create_train_val_split(x_train, y_train, val_size=0.2, random_state=42):
    """
    Splits the training data into training and validation sets with stratified sampling.
    """
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train
    )
    return x_train, x_val, y_train, y_val

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=20):
    """
    Trains the Transformer model and evaluates it on a validation set.
    """
    model.train()
    epoch_losses = []
    epoch_accuracies = []
    val_losses = []

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == batch_y).sum().item()
            total += batch_y.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        epoch_losses.append(avg_loss)
        epoch_accuracies.append(accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")

        val_loss = None
        if val_loader is not None:
            val_metrics = evaluate_model(model, val_loader, device, criterion=criterion, return_metrics=True)
            val_loss = val_metrics["loss"]  
            val_losses.append(val_loss)

            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_metrics['accuracy'] * 100:.2f}%")

            if scheduler:
                scheduler.step(val_loss)  

    final_val_loss = float(val_losses[-1]) if val_losses else float('inf') 
    return epoch_losses, epoch_accuracies, final_val_loss

def evaluate_model(model, test_loader, device, criterion=None, return_metrics=False):
    """
    Evaluates the model on the test dataset and computes classification metrics.
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    total_loss = 0.0  
    total_samples = 0  

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)

            if outputs.shape[1] == 2:  
                all_probs.append(probs[:, 1].cpu())  # Use class 1 probability
            else:
                all_probs.append(probs[:, 0].cpu())  # Use class 0 probability (fallback)

            _, predicted = torch.max(outputs, 1)
            all_preds.append(predicted.cpu())
            all_labels.append(batch_y.cpu())
            
            if criterion is not None:
                loss = criterion(outputs, batch_y)  
                total_loss += loss.item() * batch_y.size(0)
                total_samples += batch_y.size(0)

    all_preds = torch.cat(all_preds)
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    accuracy = (all_preds == all_labels).float().mean().item()
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError as e:
        print(f"AUC calculation error: {e}")
        auc = 0.0

    specificity = []
    for i in range(len(conf_matrix)):
        tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
        fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    avg_specificity = sum(specificity) / len(specificity)
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    print(f"Test Accuracy: {accuracy * 100:.2f}% | Validation Loss: {avg_loss:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Specificity: {avg_specificity:.4f}, AUC: {auc:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": avg_specificity,
        "auc": auc,
        "confusion_matrix": conf_matrix
    }

    return metrics if return_metrics else avg_loss


def split_data(x_train, y_train, test_size=0.2, random_state=42):
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=test_size, random_state=random_state)
    return x_train, x_val, y_train, y_val
