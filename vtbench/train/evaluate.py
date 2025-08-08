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

# ========================
# Single Chart Model Evaluation
# ========================
def evaluate_single_chart_model(model, test_loader):
    """Evaluation for single chart models"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in test_loader:
            images, labels = move_to_device(batch, device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = F.softmax(outputs, dim=1)[:, 1] if outputs.shape[1] == 2 else F.softmax(outputs, dim=1).max(dim=1)[0]
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return _calculate_metrics(all_labels, all_preds, all_probs, total_loss, len(test_loader))

# ========================
# Two-Branch Model Evaluation
# ========================
def evaluate_two_branch_model(model, test_chart_loader, test_num_loader):
    """Evaluation specifically for TwoBranchModel (chart + numerical)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    # Create iterators
    chart_iter = iter(test_chart_loader)
    num_iter = iter(test_num_loader)
    
    with torch.no_grad():
        while True:
            try:
                # Get batches from both loaders
                chart_batch = next(chart_iter)
                num_batch = next(num_iter)
                
                # Unpack and move to device
                chart_imgs, chart_labels = chart_batch
                chart_imgs = chart_imgs.to(device)
                chart_labels = chart_labels.to(device)
                
                num_features, _ = num_batch
                num_features = num_features.to(device)
                
                # TwoBranchModel expects (chart_tensor, num_tensor)
                outputs = model((chart_imgs, num_features))
                labels = chart_labels
                
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                probs = F.softmax(outputs, dim=1)[:, 1] if outputs.shape[1] == 2 else F.softmax(outputs, dim=1).max(dim=1)[0]
                preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
            except StopIteration:
                break

    num_batches = min(len(test_chart_loader), len(test_num_loader))
    return _calculate_metrics(all_labels, all_preds, all_probs, total_loss, num_batches)

# ========================
# Multi-Chart Model Evaluation
# ========================
def evaluate_multichart_model(model, test_chart_loaders, test_num_loader=None):
    """Evaluation for MultiChartModel (multiple charts + optional numerical)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    # Create iterators for all chart loaders
    chart_iters = [iter(loader) for loader in test_chart_loaders]
    num_iter = iter(test_num_loader) if test_num_loader else None
    
    with torch.no_grad():
        while True:
            try:
                # Get batches from all chart loaders
                chart_batches = [next(chart_iter) for chart_iter in chart_iters]
                
                # Process chart data
                chart_imgs = []
                labels = None
                for chart_batch in chart_batches:
                    imgs, lbls = chart_batch
                    imgs = imgs.to(device)
                    lbls = lbls.to(device)
                    chart_imgs.append(imgs)
                    if labels is None:
                        labels = lbls  # Use labels from first chart
                
                if test_num_loader:
                    # Get numerical batch
                    num_batch = next(num_iter)
                    num_features, _ = num_batch
                    num_features = num_features.to(device)
                    
                    # MultiChartModel with numerical expects ([chart_tensors], num_tensor)
                    outputs = model((chart_imgs, num_features))
                else:
                    # MultiChartModel without numerical expects [chart_tensors]
                    outputs = model(chart_imgs)
                
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                probs = F.softmax(outputs, dim=1)[:, 1] if outputs.shape[1] == 2 else F.softmax(outputs, dim=1).max(dim=1)[0]
                preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
            except StopIteration:
                break

    num_batches = len(test_chart_loaders[0])
    if test_num_loader:
        num_batches = min(num_batches, len(test_num_loader))
    
    return _calculate_metrics(all_labels, all_preds, all_probs, total_loss, num_batches)

# ========================
# Helper function for metrics calculation
# ========================
def _calculate_metrics(all_labels, all_preds, all_probs, total_loss, num_batches):
    """Calculate evaluation metrics"""
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
        "loss": total_loss / num_batches,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
        "specificity": specificity
    }

# ========================
# Main evaluation dispatcher (for backwards compatibility)
# ========================
def evaluate_model(model, test_chart_loaders, test_num_loader=None, model_type=None):
    """
    Main evaluation function that dispatches to the appropriate specialized function
    
    Args:
        model: The trained model
        test_chart_loaders: Single loader or list of loaders
        test_num_loader: Numerical data loader (optional)
        model_type: 'single_chart', 'two_branch', or 'multi_chart' (can be auto-detected)
    """
    
    # Auto-detect model type if not provided
    if model_type is None:
        if isinstance(test_chart_loaders, list):
            if len(test_chart_loaders) == 1 and test_num_loader is not None:
                model_type = 'two_branch'
            else:
                model_type = 'multi_chart'
        else:
            model_type = 'single_chart'
    
    # Dispatch to appropriate evaluation function
    if model_type == 'single_chart':
        return evaluate_single_chart_model(model, test_chart_loaders)
    elif model_type == 'two_branch':
        if isinstance(test_chart_loaders, list):
            chart_loader = test_chart_loaders[0]
        else:
            chart_loader = test_chart_loaders
        return evaluate_two_branch_model(model, chart_loader, test_num_loader)
    elif model_type == 'multi_chart':
        return evaluate_multichart_model(model, test_chart_loaders, test_num_loader)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")