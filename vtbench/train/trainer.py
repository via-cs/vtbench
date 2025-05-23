import torch
import torch.nn as nn
import torch.optim as optim
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

    model = get_chart_model(config['model']['chart_model'], input_channels=3, num_classes=num_classes).to(device)
    return train_standard_model(model, train_loader, val_loader, test_loader, config)

# ========================
# Two-branch model
# ========================

def train_two_branch_model(config):
    print("Training two-branch model: Chart + Numerical")
    loaders = create_dataloaders(config)
    chart_loader = loaders['train']['chart']
    num_loader = loaders['train']['numerical']

    feature_size = 64 if config['model']['chart_model'] == 'simplecnn' else 256
    chart_branch = get_chart_model(config['model']['chart_model'], 3, None)

    input_dim = next(iter(num_loader))[0].shape[1]
    num_branch = get_numerical_model(config, input_dim, feature_size)

    fusion = FusionModule(config['model']['fusion'], feature_size, num_branches=2)
    model = TwoBranchModel(chart_branch, num_branch, fusion).to(device)

    return train_multimodal_model(
        model,
        [chart_loader], num_loader,
        [loaders['test']['chart']], loaders['test']['numerical'],
        config
    )

# ========================
# Multi-chart model (with or without numerical)
# ========================

def train_multi_chart_model(config):
    print("Training multi-chart model" + (" + numerical" if config['model']['numerical_branch'] != 'none' else ""))
    loaders = create_dataloaders(config)

    chart_model = config['model']['chart_model']
    feature_size = 64 if chart_model == 'simplecnn' else 256

    train_charts = loaders['train']['chart']
    test_charts = loaders['test']['chart']
    train_numerical = loaders['train']['numerical']
    test_numerical = loaders['test']['numerical']

    branches = [get_chart_model(chart_model, 3, None) for _ in train_charts]

    if config['model']['numerical_branch'] != 'none':
        input_dim = next(iter(train_numerical))[0].shape[1]
        num_branch = get_numerical_model(config, input_dim, feature_size)
        fusion = FusionModule(config['model']['fusion'], feature_size, num_branches=len(branches) + 1)
        model = MultiChartNumericalModel(branches, num_branch, fusion).to(device)
    else:
        fusion = FusionModule(config['model']['fusion'], feature_size, num_branches=len(branches))
        model = MultiChartModel(branches, fusion).to(device)

    return train_multimodal_model(
        model,
        train_charts, train_numerical,
        test_charts, test_numerical,
        config
    )

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
            output_dim=cfg.get('output_dim', feature_size)
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

# ========================
# Multimodal trainer (shared)
# ========================

def train_multimodal_model(model, train_charts, train_numerical, test_charts, test_numerical, config):
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    patience = 10
    trigger_times = 0
    best_val_acc = 0

    def move(x): return [i.to(device) if isinstance(i, torch.Tensor) else i for i in x]

    for epoch in range(config['training']['epochs']):
        model.train()
        correct, total, running_loss = 0, 0, 0

        iterator = zip(zip(*train_charts), train_numerical) if train_numerical else zip(*train_charts)

        for batch in iterator:
            if train_numerical:
                chart_batches, num_input = batch
                chart_batches = move(chart_batches)
                num_features, labels = move(num_input)
                chart_imgs = [img for img, _ in chart_batches]
                outputs = model((chart_imgs, num_features))
            else:
                chart_batches = move(batch)
                chart_imgs = [img for img, _ in chart_batches]
                labels = chart_batches[0][1]
                outputs = model(chart_imgs)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        acc = 100 * correct / total
        print(f"[Epoch {epoch+1}] Loss: {running_loss/len(train_charts[0]):.4f}, Accuracy: {acc:.2f}%")
        scheduler.step(running_loss)

        if acc > best_val_acc:
            best_val_acc = acc
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping.")
                break

    return model
