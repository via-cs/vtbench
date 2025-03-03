import torch
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from num.CNN_utils import TimeSeriesImageDatasetMC, NumericalDataset, augmentation_transforms
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

from torch.utils.data.sampler import SubsetRandomSampler

def create_dataloaders(X_train, y_train, X_test, y_test, config, seed=42):
    """
    Create dataloaders for each chart type configuration while ensuring that training indices match across all configurations.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataloaders = {}
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

   
    total_indices = np.arange(len(y_train))
    np.random.shuffle(total_indices)
    train_sampler = SubsetRandomSampler(total_indices)

    val_size = int(0.2 * len(y_test))
    test_indices, val_indices = train_test_split(
        np.arange(len(y_test)),
        test_size=val_size,
        stratify=y_test,
        random_state=seed
    )

    if config['numerical_processing']['method'] != "none":
        numerical_train_dataset = NumericalDataset(X_train, y_train)
        numerical_test_dataset = NumericalDataset(X_test, y_test)
        
        numerical_val_dataset = Subset(numerical_test_dataset, val_indices)
        numerical_final_test_dataset = Subset(numerical_test_dataset, test_indices)
        
        dataloaders['numerical'] = {
            'train': DataLoader(
                numerical_train_dataset,
                batch_size=config['batch_size'],
                sampler=train_sampler,
                num_workers=4,
                pin_memory=True
            ),
            'val': DataLoader(
                numerical_val_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=4,
                pin_memory=True
            ),
            'test': DataLoader(
                numerical_final_test_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        }

    # chart type loaders
    for model_config in config['models']:
        combo_key = model_config['combo_key']
        logging.info(f"Creating DataLoader for {combo_key}")

        train_dataset = TimeSeriesImageDatasetMC(
            dataset_name=config['dataset_name'],
            time_series_data=X_train,
            labels=y_train,
            split='train',
            config = config,
            transform=transform,
            chart_type=model_config['chart_type'],
            color_mode=model_config.get('color_mode'),
            label_mode=model_config.get('label_mode')
        )

        test_dataset = TimeSeriesImageDatasetMC(
            dataset_name=config['dataset_name'],
            time_series_data=X_test,
            labels=y_test,
            split='test',
            config=config,
            transform=transform,
            chart_type=model_config['chart_type'],
            color_mode=model_config.get('color_mode'),
            label_mode=model_config.get('label_mode')
        )
        
        val_dataset = Subset(test_dataset, val_indices)
        final_test_dataset = Subset(test_dataset, test_indices)

        dataloaders[combo_key] = {
            'train': DataLoader(
                train_dataset,
                batch_size=config['batch_size'],
                sampler=train_sampler,
                num_workers=4,
                pin_memory=True
            ),
            'val': DataLoader(
                val_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=4,
                pin_memory=True
            ),
            'test': DataLoader(
                final_test_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        }
    
    return dataloaders

def print_class_distribution(labels):
    """Print distribution of classes in the dataset"""
    unique, counts = np.unique(labels, return_counts=True)
    print("\nClass distribution:")
    for class_idx, count in zip(unique, counts):
        print(f"Class {class_idx}: {count} samples ({count/len(labels)*100:.2f}%)")

def get_class_weights(train_labels):
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    counts = torch.bincount(train_labels)
    weights = 1.0 / counts.float()
    weights = weights / weights.sum()  
    return weights * 2


def train(model, dataloaders, config, device, train_labels):
    """
    Training function that handles multiple chart configurations
    
    """
    print("\nInitial class distribution in training data:")
    print_class_distribution(train_labels)

    class_weights = get_class_weights(train_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight = class_weights)
    
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
    
   
    chart_configs = {
        key.split('_')[0]: key 
        for key in dataloaders.keys() 
        if key != 'numerical'
    }

    train_loaders = {chart: dataloaders[chart_configs[chart]]['train'] for chart in chart_configs.keys()}
    val_loaders = {chart: dataloaders[chart_configs[chart]]['val'] for chart in chart_configs.keys()}
    test_loaders = {chart: dataloaders[chart_configs[chart]]['test'] for chart in chart_configs.keys()}

    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
         
        train_iters = {chart: iter(train_loaders[chart]) for chart in chart_configs.keys()}
        
        if config['numerical_processing']['method'] != "none":
            numerical_train_iter = iter(dataloaders['numerical']['train'])
        
        num_batches = min([len(loader) for loader in train_loaders.values()])
        
        for _ in range(num_batches):
            try:
               
                bar_batch = next(train_iters['bar'])
                line_batch = next(train_iters['line'])
                area_batch = next(train_iters['area'])
                scatter_batch = next(train_iters['scatter'])
                
              
                bar_imgs = bar_batch[0].to(device)
                line_imgs = line_batch[0].to(device)
                area_imgs = area_batch[0].to(device)
                scatter_imgs = scatter_batch[0].to(device)
               
                labels = bar_batch[1].to(device)

                if config['numerical_processing']['method'] != "none":
                    try:
                        numerical_batch = next(numerical_train_iter)
                        numerical_features = numerical_batch[0].to(device)
                    except StopIteration:
                        numerical_train_iter = iter(dataloaders['numerical']['train'])
                        numerical_batch = next(numerical_train_iter)
                        numerical_features = numerical_batch[0].to(device)
                else:
                    numerical_features = None

                               
                optimizer.zero_grad()
                outputs = model(bar_imgs, line_imgs, area_imgs, scatter_imgs, numerical_features)
                loss = criterion(outputs, labels)
                
                
                loss.backward()
                optimizer.step()
                
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
            except StopIteration:
                break
        
       
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        val_iters = {chart: iter(val_loaders[chart]) for chart in chart_configs.keys()}

        if config['numerical_processing']['method'] != "none":
            numerical_val_iter = iter(dataloaders['numerical']['val'])

        num_val_batches = min([len(loader) for loader in val_loaders.values()])
        
        with torch.no_grad():
            for _ in range(num_val_batches):
                try:
                    # Get batch from each loader
                    bar_batch = next(val_iters['bar'])
                    line_batch = next(val_iters['line'])
                    area_batch = next(val_iters['area'])
                    scatter_batch = next(val_iters['scatter'])
                    
                    bar_imgs = bar_batch[0].to(device)
                    line_imgs = line_batch[0].to(device)
                    area_imgs = area_batch[0].to(device)
                    scatter_imgs = scatter_batch[0].to(device)
                    labels = bar_batch[1].to(device)

                    if config['numerical_processing']['method'] != "none":
                        try:
                            numerical_batch = next(numerical_val_iter)
                            numerical_features = numerical_batch[0].to(device)
                        except StopIteration:
                            numerical_val_iter = iter(dataloaders['numerical']['val'])
                            numerical_batch = next(numerical_val_iter)
                            numerical_features = numerical_batch[0].to(device)
                    else:
                        numerical_features = None
                    
                    outputs = model(bar_imgs, line_imgs, area_imgs, scatter_imgs, numerical_features)
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
    

def test(model, dataloaders, device, config):
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
    
    # Get test loaders for each chart type, excluding numerical data loader if present
    chart_configs = {
        key.split('_')[0]: key 
        for key in dataloaders.keys() 
        if key != 'numerical'
    }

    test_loaders = {chart: dataloaders[chart_configs[chart]]['test'] for chart in chart_configs.keys()}
    test_iters = {chart: iter(test_loaders[chart]) for chart in chart_configs.keys()}

    numerical_method = config['numerical_processing']['method']
    if numerical_method != "none":
        try:
            numerical_test_iter = iter(dataloaders['numerical']['test'])
        except KeyError:
            raise ValueError("Numerical processing is enabled but numerical test loader not found")
    
    num_test_batches = min([len(loader) for loader in test_loaders.values()])

    
    with torch.no_grad():
        for batch_idx in range(num_test_batches):
            try:
                # Get batch from each loader
                bar_batch = next(test_iters['bar'])
                line_batch = next(test_iters['line'])
                area_batch = next(test_iters['area'])
                scatter_batch = next(test_iters['scatter'])
                
                # Move data to devices
                bar_imgs = bar_batch[0].to(device)
                line_imgs = line_batch[0].to(device)
                area_imgs = area_batch[0].to(device)
                scatter_imgs = scatter_batch[0].to(device)
                labels = bar_batch[1].to(device)  

                if numerical_method != "none":
                    try:
                        numerical_batch = next(numerical_test_iter)
                        numerical_features = numerical_batch[0].to(device)
                    except StopIteration:
                        numerical_test_iter = iter(dataloaders['numerical']['test'])
                        numerical_batch = next(numerical_test_iter)
                        numerical_features = numerical_batch[0].to(device)
                else:
                    numerical_features = None

                
                # Forward pass
                outputs = model(bar_imgs, line_imgs, area_imgs, scatter_imgs, numerical_features)
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