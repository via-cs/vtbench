import torch
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from num.CNN_utils import TimeSeriesImageDatasetMC
import logging
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)
import logging
import seaborn as sns
import matplotlib.pyplot as plt

def shuffle_indices(length, seed):
    """Create shuffled indices"""
    indices = np.arange(length)
    np.random.seed(seed)
    np.random.shuffle(indices)
    return indices

def stratified_split(dataset, labels, val_size):
    """Perform stratified split of the test dataset"""
    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=val_size,
        stratify=labels,
        random_state=42
    )
    return Subset(dataset, val_idx), Subset(dataset, train_idx)

def print_class_distribution(labels):
    """Print distribution of classes in the dataset"""
    unique, counts = np.unique(labels, return_counts=True)
    print("\nClass distribution:")
    for class_idx, count in zip(unique, counts):
        print(f"Class {class_idx}: {count} samples ({count/len(labels)*100:.2f}%)")

def create_dataloaders(X_train, y_train, X_test, y_test, config, seed=42):
    """Create dataloaders with monitoring"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Print initial class distributions
    print("\nTraining set class distribution:")
    print_class_distribution(y_train)
    print("\nTest set class distribution:")
    print_class_distribution(y_test)
    
    dataloaders = {}
    total_samples = {"train": 0, "val": 0, "test": 0}
    
    # Shuffle indices
    shuffled_train_indices = shuffle_indices(len(y_train), seed)
    shuffled_test_indices = shuffle_indices(len(y_test), seed)
    
    # Apply shuffling
    X_train = X_train[shuffled_train_indices]
    y_train = y_train[shuffled_train_indices]
    X_test = X_test[shuffled_test_indices]
    y_test = y_test[shuffled_test_indices]

    # Create dataloaders for each model configuration
    for model_config in config['models']:
        # Extract configuration parameters
        chart_type = model_config['chart_type']
        color_mode = model_config.get('color_mode', 'color')
        label_mode = model_config.get('label_mode', 'with_label')
        scatter_mode = model_config.get('scatter_mode', '')
        bar_mode = model_config.get('bar_mode', '')

        # Create combo key
        if chart_type == 'bar':
            combo_key = f"{chart_type}_{bar_mode}_{color_mode}_{label_mode}"
        elif chart_type == 'scatter':
            combo_key = f"{chart_type}_{scatter_mode}_{color_mode}_{label_mode}"
        else:
            combo_key = f"{chart_type}_{color_mode}_{label_mode}"

        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        # Create datasets
        train_dataset = TimeSeriesImageDatasetMC(
            dataset_name=config['dataset_name'],
            time_series_data=X_train,
            labels=y_train,
            split='train',
            transform=transform,
            chart_type=chart_type,
            color_mode=color_mode,
            label_mode=label_mode,
            scatter_mode=scatter_mode,
            bar_mode=bar_mode
        )

        test_dataset = TimeSeriesImageDatasetMC(
            dataset_name=config['dataset_name'],
            time_series_data=X_test,
            labels=y_test,
            split='test',
            transform=transform,
            chart_type=chart_type,
            color_mode=color_mode,
            label_mode=label_mode,
            scatter_mode=scatter_mode,
            bar_mode=bar_mode
        )

        # Create validation split
        val_size = int(0.2 * len(test_dataset))
        val_dataset, final_test_dataset = stratified_split(test_dataset, y_test, val_size/len(test_dataset))

        # Create dataloaders with smaller batch size
        train_loader = DataLoader(
            train_dataset,
            batch_size=16,  # Reduced batch size
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        test_loader = DataLoader(
            final_test_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        dataloaders[combo_key] = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
        
        # Update total samples
        total_samples["train"] += len(train_dataset)
        total_samples["val"] += len(val_dataset)
        total_samples["test"] += len(final_test_dataset)

    return dataloaders

def train(model, dataloaders, config, device, train_labels):
    """Training function with improved stability"""
    print("\nInitial class distribution in training data:")
    print_class_distribution(train_labels)
    
    # More aggressive class weights for better balance
    counts = torch.bincount(torch.tensor(train_labels))
    total = len(train_labels)
    inv_freq = total / (counts * 2)  # Multiply minority class weight by 2
    class_weights = inv_freq / inv_freq.sum()
    class_weights = class_weights.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # More stable optimizer settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0005,
        weight_decay=0.01,
        betas=(0.9, 0.99)
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Get loaders for each chart type
    chart_configs = {config_key.split('_')[0]: config_key 
                    for config_key in dataloaders.keys()}

    train_loaders = {
        'bar': dataloaders[chart_configs['bar']]['train'],
        'line': dataloaders[chart_configs['line']]['train'],
        'area': dataloaders[chart_configs['area']]['train'],
        'scatter': dataloaders[chart_configs['scatter']]['train']
    }
    
    val_loaders = {
        'bar': dataloaders[chart_configs['bar']]['val'],
        'line': dataloaders[chart_configs['line']]['val'],
        'area': dataloaders[chart_configs['area']]['val'],
        'scatter': dataloaders[chart_configs['scatter']]['val']
    }
    
    # Training loop
    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        epoch_predictions = []
        
        # Create iterators
        train_iters = {k: iter(v) for k, v in train_loaders.items()}
        num_batches = min(len(loader) for loader in train_loaders.values())
        
        # Training batch loop
        for batch_idx in range(num_batches):
            optimizer.zero_grad()
            batch_loss = 0
            
            try:
                # Process each chart type
                bar_batch = next(train_iters['bar'])
                line_batch = next(train_iters['line'])
                area_batch = next(train_iters['area'])
                scatter_batch = next(train_iters['scatter'])
                
                # Move data to device
                bar_imgs = bar_batch[0].to(device)
                line_imgs = line_batch[0].to(device)
                area_imgs = area_batch[0].to(device)
                scatter_imgs = scatter_batch[0].to(device)
                labels = bar_batch[1].to(device)
                
                # Forward pass with gradient accumulation
                outputs = model(bar_imgs, line_imgs, area_imgs, scatter_imgs)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Metrics
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                epoch_predictions.extend(predicted.cpu().numpy())
                
                # Monitor predictions periodically
                if batch_idx % 50 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"\nBatch {batch_idx} - LR: {current_lr:.6f}")
                    print("Prediction distribution:")
                    print_class_distribution(np.array(predicted.cpu()))
                
            except StopIteration:
                break
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_predictions = []
        
        val_iters = {k: iter(v) for k, v in val_loaders.items()}
        num_val_batches = min(len(loader) for loader in val_loaders.values())
        
        with torch.no_grad():
            for _ in range(num_val_batches):
                try:
                    # Process validation data
                    bar_batch = next(val_iters['bar'])
                    line_batch = next(val_iters['line'])
                    area_batch = next(val_iters['area'])
                    scatter_batch = next(val_iters['scatter'])
                    
                    bar_imgs = bar_batch[0].to(device)
                    line_imgs = line_batch[0].to(device)
                    area_imgs = area_batch[0].to(device)
                    scatter_imgs = scatter_batch[0].to(device)
                    labels = bar_batch[1].to(device)
                    
                    outputs = model(bar_imgs, line_imgs, area_imgs, scatter_imgs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    val_predictions.extend(predicted.cpu().numpy())
                    
                except StopIteration:
                    break
        
        # Calculate epoch metrics
        train_loss = train_loss / num_batches
        train_acc = 100. * train_correct / train_total
        val_loss = val_loss / num_val_batches
        val_acc = 100. * val_correct / val_total

        scheduler.step(val_acc)
        
        # Print metrics
        print(f'\nEpoch [{epoch+1}/{config["num_epochs"]}] '
              f'Train Loss: {train_loss:.4f} '
              f'Train Acc: {train_acc:.2f}% '
              f'Val Loss: {val_loss:.4f} '
              f'Val Acc: {val_acc:.2f}%')
        
        print("\nTraining prediction distribution:")
        print_class_distribution(np.array(epoch_predictions))
        print("\nValidation prediction distribution:")
        print_class_distribution(np.array(val_predictions))
        
        # Early stopping with minimum improvement threshold
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()  # Store in memory
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f'Early stopping after {epoch+1} epochs')
                break
    
    # Load best model state from memory before returning
    model.load_state_dict(best_model_state)
    return model



def test(model, dataloaders, device):
    """Test function with detailed metrics"""
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    criterion = nn.CrossEntropyLoss()
    
    # Get test loaders
    chart_configs = {}
    for config_key in dataloaders.keys():
        chart_type = config_key.split('_')[0]
        chart_configs[chart_type] = config_key

    test_loaders = {
        'bar': dataloaders[chart_configs['bar']]['test'],
        'line': dataloaders[chart_configs['line']]['test'],
        'area': dataloaders[chart_configs['area']]['test'],
        'scatter': dataloaders[chart_configs['scatter']]['test']
    }
    
    test_iters = {k: iter(v) for k, v in test_loaders.items()}
    num_test_batches = min([len(loader) for loader in test_loaders.values()])
    
    with torch.no_grad():
        for _ in range(num_test_batches):
            try:
                bar_batch = next(test_iters['bar'])
                line_batch = next(test_iters['line'])
                area_batch = next(test_iters['area'])
                scatter_batch = next(test_iters['scatter'])
                
                bar_imgs = bar_batch[0].to(device)
                line_imgs = line_batch[0].to(device)
                area_imgs = area_batch[0].to(device)
                scatter_imgs = scatter_batch[0].to(device)
                labels = bar_batch[1].to(device)
                
                outputs = model(bar_imgs, line_imgs, area_imgs, scatter_imgs)
                loss = criterion(outputs, labels)
                probs = torch.softmax(outputs, dim=1)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
            except StopIteration:
                break
    
    # Calculate test loss and accuracy
    test_loss = test_loss / num_test_batches
    test_accuracy = 100. * test_correct / test_total
    
    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = conf_matrix.ravel()
    
    # Calculate metrics
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)  # Sensitivity
    specificity = tn / (tn + fp)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    # Calculate ROC AUC
    fpr, tpr, _ = roc_curve(all_labels, np.array(all_probs)[:, 1])
    auc_score = auc(fpr, tpr)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Print metrics in the desired format
    print("\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    return test_loss, test_accuracy, precision, recall, specificity, f1, auc_score, conf_matrix