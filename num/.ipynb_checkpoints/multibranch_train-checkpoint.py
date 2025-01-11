import torch
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from num.CNN_utils import TimeSeriesImageDatasetMC, augmentation_transforms
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

def create_dataloaders(X_train, y_train, X_test, y_test, config, seed=42):
    """
    Create dataloaders for each chart type configuration with 20% stratified validation split
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
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
            
        logging.info(f"Creating DataLoader for combo key: {combo_key}")

        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        # Apply SMOTE if configured
        if config.get('apply_smote', False):
            logging.info(f"Applying SMOTE for {combo_key}...")
            desired_samples = config['desired_samples_per_class']
            X_resampled, y_resampled = apply_smote(X_train, y_train, desired_samples)
            logging.info(f"After SMOTE - Class distribution: {np.bincount(y_resampled)}")
        else:
            logging.info(f"Running without SMOTE for {combo_key}...")
            X_resampled, y_resampled = X_train, y_train

        # Create training dataset
        train_dataset = TimeSeriesImageDatasetMC(
            dataset_name=config['dataset_name'],
            time_series_data=X_resampled,
            labels=y_resampled,
            split='train',
            # transform=augmentation_transforms,
            transform = transform,
            chart_type=chart_type,
            color_mode=color_mode,
            label_mode=label_mode,
            scatter_mode=scatter_mode,
            bar_mode=bar_mode
        )

        # Create test dataset
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

        # Create validation and test splits (20% for validation)
        indices = np.arange(len(test_dataset))
        val_size = int(0.2 * len(indices))  # 20% of test data
        
        # Use sklearn's train_test_split for stratified splitting
        test_indices, val_indices = train_test_split(
            indices,
            test_size=val_size,
            stratify=y_test,
            random_state=seed
        )

        # Create validation and test datasets using the split indices
        val_dataset = Subset(test_dataset, val_indices)
        final_test_dataset = Subset(test_dataset, test_indices)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        test_loader = DataLoader(
            final_test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        dataloaders[combo_key] = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
        
        total_samples["train"] += len(train_dataset)
        total_samples["val"] += len(val_dataset)
        total_samples["test"] += len(final_test_dataset)

        logging.info(f"\nDataset sizes for {combo_key}:")
        logging.info(f"Training: {len(train_dataset)}")
        logging.info(f"Validation: {len(val_dataset)} (20% of test data)")
        logging.info(f"Test: {len(final_test_dataset)} (80% of test data)")

    logging.info("\nTotal samples across all configurations:")
    logging.info(f"Training: {total_samples['train']}")
    logging.info(f"Validation: {total_samples['val']}")
    logging.info(f"Test: {total_samples['test']}")

    return dataloaders

def print_class_distribution(labels):
    """Print distribution of classes in the dataset"""
    unique, counts = np.unique(labels, return_counts=True)
    print("\nClass distribution:")
    for class_idx, count in zip(unique, counts):
        print(f"Class {class_idx}: {count} samples ({count/len(labels)*100:.2f}%)")

def get_class_weights(train_labels):
    counts = torch.bincount(torch.tensor(train_labels))
    weights = 1.0 / counts.float()
    weights = weights / weights.sum()  # normalize
    return weights * 2


def train(model, dataloaders, config, device, train_labels):
    """
    Training function that handles multiple chart configurations
    Args:
        model: MultiBranchCNN model
        dataloaders: Dictionary of dataloaders for each chart configuration
        config: Configuration dictionary
        device: torch device
    """
    print("\nInitial class distribution in training data:")
    print_class_distribution(train_labels)

    class_weights = get_class_weights(train_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=10, 
        verbose=True,
        min_lr = 1e-6
    )

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Get reference to individual loaders for each chart type
    chart_configs = {}
    for config_key in dataloaders.keys():
        chart_type = config_key.split('_')[0]  # Extract chart type from combo key
        chart_configs[chart_type] = config_key

    # Get loaders for each chart type
    bar_loaders = dataloaders[chart_configs['bar']]
    line_loaders = dataloaders[chart_configs['line']]
    area_loaders = dataloaders[chart_configs['area']]
    scatter_loaders = dataloaders[chart_configs['scatter']]
    
    train_loaders = {
        'bar': bar_loaders['train'],
        'line': line_loaders['train'],
        'area': area_loaders['train'],
        'scatter': scatter_loaders['train']
    }
    
    val_loaders = {
        'bar': bar_loaders['val'],
        'line': line_loaders['val'],
        'area': area_loaders['val'],
        'scatter': scatter_loaders['val']
    }
    
    test_loaders = {
        'bar': bar_loaders['test'],
        'line': line_loaders['test'],
        'area': area_loaders['test'],
        'scatter': scatter_loaders['test']
    }
    
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Create iterators for all loaders
        train_iters = {
            'bar': iter(train_loaders['bar']),
            'line': iter(train_loaders['line']),
            'area': iter(train_loaders['area']),
            'scatter': iter(train_loaders['scatter'])
        }
        
        # Determine number of batches based on shortest loader
        num_batches = min([len(loader) for loader in train_loaders.values()])
        
        for _ in range(num_batches):
            try:
                # Get batch from each loader
                bar_batch = next(train_iters['bar'])
                line_batch = next(train_iters['line'])
                area_batch = next(train_iters['area'])
                scatter_batch = next(train_iters['scatter'])
                
                # Move data to device
                bar_imgs = bar_batch[0].to(device)
                line_imgs = line_batch[0].to(device)
                area_imgs = area_batch[0].to(device)
                scatter_imgs = scatter_batch[0].to(device)
                # Use labels from any batch (they should be the same)
                labels = bar_batch[1].to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(bar_imgs, line_imgs, area_imgs, scatter_imgs)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Calculate metrics
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
            except StopIteration:
                break
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        val_iters = {
            'bar': iter(val_loaders['bar']),
            'line': iter(val_loaders['line']),
            'area': iter(val_loaders['area']),
            'scatter': iter(val_loaders['scatter'])
        }
        
        num_val_batches = min([len(loader) for loader in val_loaders.values()])
        
        with torch.no_grad():
            for _ in range(num_val_batches):
                try:
                    # Get batch from each loader
                    bar_batch = next(val_iters['bar'])
                    line_batch = next(val_iters['line'])
                    area_batch = next(val_iters['area'])
                    scatter_batch = next(val_iters['scatter'])
                    
                    # Move data to device
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
                    
                except StopIteration:
                    break
        
        # Calculate epoch metrics
        train_loss = train_loss / num_batches
        train_acc = 100. * train_correct / train_total
        val_loss = val_loss / num_val_batches
        val_acc = 100. * val_correct / val_total
        
        # Print metrics (simplified format)
        print(f'Epoch [{epoch+1}/{config["num_epochs"]}] '
              f'Train Loss: {train_loss:.4f} '
              f'Train Acc: {train_acc:.2f}% '
              f'Val Loss: {val_loss:.4f} '
              f'Val Acc: {val_acc:.2f}%')
        
        scheduler.step(val_acc)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f'Early stopping after {epoch+1} epochs')
                break
    
    return model
    

def test(model, dataloaders, device):
    """
    Test function with specific metric outputs
    """
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    criterion = nn.CrossEntropyLoss()
    
    # Get test loaders for each chart type
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
                # Get batch from each loader
                bar_batch = next(test_iters['bar'])
                line_batch = next(test_iters['line'])
                area_batch = next(test_iters['area'])
                scatter_batch = next(test_iters['scatter'])
                
                # Move data to device
                bar_imgs = bar_batch[0].to(device)
                line_imgs = line_batch[0].to(device)
                area_imgs = area_batch[0].to(device)
                scatter_imgs = scatter_batch[0].to(device)
                labels = bar_batch[1].to(device)
                
                # Forward pass
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
    
    # Calculate metrics
    test_loss = test_loss / num_test_batches
    test_accuracy = 100. * test_correct / test_total
    
    # Convert to numpy arrays
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
    fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
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
