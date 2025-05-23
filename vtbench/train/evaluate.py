import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import torch.nn.functional as F
import torch.nn as nn 

def move_to_device(tensors, device):
    if isinstance(tensors, (tuple, list)):
        return [t.to(device) for t in tensors]
    return tensors.to(device)


def evaluate_model(model, test_chart_loaders, test_num_loader=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    if isinstance(test_chart_loaders, list):
        iterator = zip(*test_chart_loaders)
    else:
        iterator = test_chart_loaders

    with torch.no_grad():
        for batch in iterator:
            if isinstance(test_chart_loaders, list):
                batch = [move_to_device(b, device) for b in batch]
                chart_imgs = [img for img, _ in batch]
                labels = batch[0][1]
                if test_num_loader:
                    num_input, _ = move_to_device(next(iter(test_num_loader)), device)
                    outputs = model((chart_imgs, num_input))
                else:
                    outputs = model(chart_imgs)
            else:
                images, labels = move_to_device(batch, device)
                outputs = model(images)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = F.softmax(outputs, dim=1)[:, 1] if outputs.shape[1] == 2 else F.softmax(outputs, dim=1).max(dim=1)[0]
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary' if len(set(all_labels)) == 2 else 'macro')
    recall = recall_score(all_labels, all_preds, average='binary' if len(set(all_labels)) == 2 else 'macro')
    f1 = f1_score(all_labels, all_preds, average='binary' if len(set(all_labels)) == 2 else 'macro')

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = float('nan')

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel() if len(set(all_labels)) == 2 else [0, 0, 0, 0]
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "loss": total_loss / len(all_preds),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
        "specificity": specificity
    }
