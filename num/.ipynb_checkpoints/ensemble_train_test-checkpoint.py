import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score, confusion_matrix
from imblearn.metrics import specificity_score

def train_model_ensemble(model, train_loader, val_loader, num_epochs=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    best_val_accuracy = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, Training Accuracy: {train_accuracy:.2f}%")

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        scheduler.step(val_loss)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")

def test_model_ensemble(model_or_preds, test_loader, return_raw=False):
    y_true = []
    y_pred = []
    y_probs = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(model_or_preds, torch.nn.Module):
        model = model_or_preds
        model.eval()
        model.to(device)

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(torch.argmax(probs, dim=1).cpu().numpy())
                y_probs.extend(probs.cpu().numpy())
    else:
        for idx, (_, label) in enumerate(test_loader):
            y_true.append(label)
            preds = np.array(model_or_preds[idx])  
            if preds.ndim == 1:
                preds = preds.reshape(1, -1)  
            y_probs.extend(preds)
            y_pred.extend(np.argmax(preds, axis=1))

    # Calculate metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, [p[1] for p in y_probs])  
    conf_matrix = confusion_matrix(y_true, y_pred)
    test_accuracy = (np.array(y_true) == np.array(y_pred)).mean() * 100

    test_loss = -np.mean([np.log(p[label]) for p, label in zip(y_probs, y_true)])

    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp)

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'specificity': specificity,
        'confusion_matrix': conf_matrix.tolist()  
    }

    if return_raw:
        return y_probs, metrics
    else:
        return metrics

