import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from vtbench.data.loader import create_dataloaders
from vtbench.models.numerical.fcn import NumericalFCN
from vtbench.models.numerical.transformer import NumericalTransformer
from vtbench.models.numerical.oscnn import NumericalOSCNN
from vtbench.models.multimodal.one_chart_numerical import TwoBranchModel
from vtbench.models.multimodal.multi_chart import MultiChartModel
from vtbench.models.multimodal.multi_chart_numerical import MultiChartNumericalModel
from vtbench.models.multimodal.fusion import FusionModule
from vtbench.train.factory import get_chart_model
import os
import yaml

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# ========================
# Main controller
# ========================

def train_model(config):
    model_type = config['model']['type']
    
    if model_type == 'single_modal_chart':
        return train_single_chart_model(config)
    
    elif model_type == 'two_branch':
        return train_two_branch_model(config)
    
    elif model_type == 'multi_modal_chart':
        return train_multi_chart_model(config)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# ========================
# Single chart model
# ========================

def train_single_chart_model(config):
    print(f"Training single chart model: {config['model']['chart_model']}")
    loaders = create_dataloaders(config)
    train_loader = loaders['train']['chart']
    val_loader = loaders['val']['chart']
    test_loader = loaders['test']['chart']

    labels = [label for _, label in train_loader.dataset]
    num_classes = len(set(labels))
    pretrained = config['model'].get('pretrained', False)

    model = get_chart_model(config['model']['chart_model'], input_channels=3, num_classes=num_classes, pretrained=pretrained).to(device)
    return train_standard_model(model, train_loader, val_loader, test_loader, config)

# ========================
# Two-branch model 
# ========================

import torch.nn as nn

def train_two_branch_model(config):
    print("Training two-branch model: Chart + Numerical")
    loaders = create_dataloaders(config)

    # Handle case where chart might be a list of loaders
    chart_data = loaders['train']['chart']
    if isinstance(chart_data, list):
        chart_loader = chart_data[0]  # Take the first (and should be only) chart loader
        test_chart_loader = loaders['test']['chart'][0]
    else:
        chart_loader = chart_data
        test_chart_loader = loaders['test']['chart']

    num_loader = loaders['train']['numerical']
    test_num_loader = loaders['test']['numerical']

    chart_model_name = config['model']['chart_model']
    pretrained = config['model'].get('pretrained', False)
    fusion_mode = config['model']['fusion']
    num_classes = int(config['model']['num_classes'])

    # Detect feature size from the actual backbone (e.g., 512 for ResNet-18)
    tmp_branch = get_chart_model(chart_model_name, input_channels=3, num_classes=None, pretrained=pretrained)
    feature_size = getattr(tmp_branch, "feature_dim", 256)

    # Build branches
    chart_branch = get_chart_model(chart_model_name, input_channels=3, num_classes=None, pretrained=pretrained)

    # Numerical branch: force transformer output_dim to match feature_size
    input_dim = next(iter(num_loader))[0].shape[1]
    cfg = config['model'].setdefault('transformer_config', {})
    cfg['output_dim'] = feature_size
    num_branch = get_numerical_model(config, input_dim, feature_size)

    # Fusion + model
    fusion = FusionModule(fusion_mode, feature_size, num_branches=2)
    model = TwoBranchModel(chart_branch, num_branch, fusion).to(device)

    # ---- Classifier dim depends on fusion mode ----
    if fusion_mode == 'concat':
        fused_dim = feature_size * 2
    elif fusion_mode == 'weighted_sum':
        fused_dim = feature_size
    else:
        raise ValueError(f"Unsupported fusion mode: {fusion_mode}")

    model.classifier = nn.Linear(fused_dim, num_classes).to(device)

    return train_two_branch_multimodal(
        model,
        chart_loader, num_loader,
        test_chart_loader, test_num_loader,
        config
    )

# ========================
# Multi-chart model (NEW SEPARATE FUNCTION)
# ========================

def train_multi_chart_model(config):
    print("Training multi-chart model" + (" + numerical" if config['model']['numerical_branch'] != 'none' else ""))
    loaders = create_dataloaders(config)

    chart_model = config['model']['chart_model']
    pretrained = config['model'].get('pretrained', False)
    fusion_mode = config['model']['fusion']
    num_classes = int(config['model']['num_classes'])

    # Detect per-branch feature size from the chosen backbone
    tmp_branch = get_chart_model(chart_model, input_channels=3, num_classes=None, pretrained=pretrained)
    feature_size = getattr(tmp_branch, "feature_dim", 256)

    train_charts = loaders['train']['chart']
    test_charts = loaders['test']['chart']

    # Build one visual branch per chart DataLoader
    branches = [get_chart_model(chart_model, 3, None, pretrained=pretrained) for _ in train_charts]
    num_visual = len(branches)

    # Numerical branch?
    has_numerical = config['model']['numerical_branch'] != 'none'
    train_numerical = loaders['train']['numerical'] if has_numerical else None
    test_numerical = loaders['test']['numerical'] if has_numerical else None
    num_branches = num_visual + (1 if has_numerical and train_numerical is not None else 0)

    if has_numerical and train_numerical is not None:
        # Determine numerical input dimension
        try:
            sample_batch = next(iter(train_numerical))
            input_dim = sample_batch[0].shape[1]
        except (StopIteration, IndexError, AttributeError) as e:
            raise ValueError(f"Could not determine input dimension from numerical data: {e}")

        # Force transformer output_dim to match visual feature_size
        cfg = config['model'].setdefault('transformer_config', {})
        cfg['output_dim'] = feature_size
        num_branch = get_numerical_model(config, input_dim, feature_size)

        fusion = FusionModule(fusion_mode, feature_size, num_branches=num_branches)
        model = MultiChartNumericalModel(branches, num_branch, fusion).to(device)
        print(f"Created MultiChartNumericalModel with {num_visual} chart branches + numerical branch")
    else:
        fusion = FusionModule(fusion_mode, feature_size, num_branches=num_visual)
        model = MultiChartModel(branches, fusion).to(device)
        train_numerical = None
        test_numerical = None
        print(f"Created MultiChartModel with {num_visual} chart branches (no numerical)")

    # ---- Classifier dim depends on fusion mode ----
    if fusion_mode == 'concat':
        fused_dim = feature_size * num_branches
    elif fusion_mode == 'weighted_sum':
        fused_dim = feature_size
    else:
        raise ValueError(f"Unsupported fusion mode: {fusion_mode}")

    model.classifier = nn.Linear(fused_dim, num_classes).to(device)

    return train_multichart_multimodal(
        model,
        train_charts, train_numerical,
        test_charts, test_numerical,
        config
    )


def train_multichart_multimodal(model, chart_loaders, num_loader, test_chart_loaders, test_num_loader, config):
    """Training function specifically for MultiChartModel (multiple charts + optional numerical)"""
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    patience = 10
    trigger_times = 0
    best_val_acc = 0

    for epoch in range(config['training']['epochs']):
        # === Training ===
        model.train()
        train_correct, train_total, train_loss = 0, 0, 0

        # Create iterators for all chart loaders
        chart_iters = [iter(loader) for loader in chart_loaders]
        num_iter = iter(num_loader) if num_loader else None
        
        # Get epoch length
        epoch_length = len(chart_loaders[0])
        if num_loader:
            epoch_length = min(epoch_length, len(num_loader))

        for batch_idx in range(epoch_length):
            optimizer.zero_grad()
            
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
                
                if num_loader:
                    # Get numerical batch
                    num_batch = next(num_iter)
                    num_features, _ = num_batch
                    num_features = num_features.to(device)
                    
                    # MultiChartModel with numerical expects ([chart_tensors], num_tensor)
                    outputs = model((chart_imgs, num_features))
                else:
                    # MultiChartModel without numerical expects [chart_tensors]
                    outputs = model(chart_imgs)
                
            except StopIteration:
                break
                
            # Backpropagation
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        train_acc = 100 * train_correct / train_total

        # === Validation ===
        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0
        
        # Create validation iterators
        val_chart_iters = [iter(loader) for loader in test_chart_loaders]
        val_num_iter = iter(test_num_loader) if test_num_loader else None
        
        val_length = len(test_chart_loaders[0])
        if test_num_loader:
            val_length = min(val_length, len(test_num_loader))
        
        with torch.no_grad():
            for batch_idx in range(val_length):
                try:
                    # Get validation batches from all chart loaders
                    val_chart_batches = [next(val_chart_iter) for val_chart_iter in val_chart_iters]
                    
                    # Process validation chart data
                    val_chart_imgs = []
                    val_labels = None
                    for val_chart_batch in val_chart_batches:
                        imgs, lbls = val_chart_batch
                        imgs = imgs.to(device)
                        lbls = lbls.to(device)
                        val_chart_imgs.append(imgs)
                        if val_labels is None:
                            val_labels = lbls  # Use labels from first chart
                    
                    if test_num_loader:
                        # Get validation numerical batch
                        val_num_batch = next(val_num_iter)
                        val_num_features, _ = val_num_batch
                        val_num_features = val_num_features.to(device)
                        
                        # Forward pass
                        val_outputs = model((val_chart_imgs, val_num_features))
                    else:
                        # Forward pass without numerical
                        val_outputs = model(val_chart_imgs)
                    
                    loss = criterion(val_outputs, val_labels)
                    val_loss += loss.item()
                    val_correct += (val_outputs.argmax(dim=1) == val_labels).sum().item()
                    val_total += val_labels.size(0)
                    
                except StopIteration:
                    break
        
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0

        print(f"[Epoch {epoch + 1}] Train Loss: {train_loss / epoch_length:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss / val_length:.4f}, Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping.")
                break

    return model

# ========================
# Numerical model factory
# ========================

def get_numerical_model(config, input_dim, feature_size):
    numerical_model = config['model']['numerical_branch']
    if numerical_model == 'fcn':
        return NumericalFCN(input_dim=input_dim, output_dim=feature_size)
    elif numerical_model == 'transformer':
        cfg = config['model'].get('transformer_config', {})
        return NumericalTransformer(
            input_dim=input_dim,
            hidden_dim=cfg.get('hidden_dim', 128),
            num_heads=cfg.get('num_heads', 4),
            num_layers=cfg.get('num_layers', 2),
            dropout=cfg.get('dropout', 0.1),
            output_dim=feature_size
        )
    elif numerical_model == 'oscnn':
        return NumericalOSCNN(output_dim=feature_size)
    else:
        raise ValueError(f"Unsupported numerical branch type: {numerical_model}")

# ========================
# Simple chart model trainer
# ========================

def train_standard_model(model, train_loader, val_loader, test_loader, config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    patience, trigger_times, best_val_acc = 10, 0, 0

    for epoch in range(config['training']['epochs']):
        # === Train ===
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total

        # === Validate ===
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping.")
                break

    return model