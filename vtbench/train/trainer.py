import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from vtbench.data.loader import create_dataloaders
from vtbench.models.numerical.fcn import NumericalFCN
from vtbench.models.numerical.transformer import NumericalTransformer
from vtbench.models.numerical.oscnn import NumericalOSCNN
from vtbench.models.multimodal.one_chart_numerical import TwoBranchModel
from vtbench.models.multimodal.multi_chart import MultiChartModel
from vtbench.models.multimodal.multi_chart_numerical import MultiChartNumericalModel
from vtbench.models.multimodal.fusion import FusionModule
from vtbench.models.chart_models.factory import get_chart_model  # <- you need to define this factory

import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# ========================
# Controller: Main trainer
# ========================

def train_model(config):
    model_type = config['model']['type']
    if model_type == 'single_modal_chart':
        return train_single_chart_model(config)
    elif model_type == 'two_branch':
        return train_two_branch_model(config)
    elif model_type == 'multi_modal_chart':
        return train_multi_chart_model(config)
    elif model_type == 'multi_modal_chart_numerical':
        return train_multi_chart_numerical_model(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# -----------------------
# Single chart
# -----------------------
def train_single_chart_model(config):
    print(f"Training a single chart model: {config['model']['chart_model']}")
    loaders = create_dataloaders(config)
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']

    labels = [label for _, label in train_loader.dataset]
    num_classes = len(set(labels))

    model = get_chart_model(config['model']['chart_model'], input_channels=3, num_classes=num_classes).to(device)
    return train_standard_model(model, train_loader, val_loader, test_loader, config)


# -----------------------
# Two-branch (chart + numerical)
# -----------------------

def train_two_branch_model(config):
    print("Training a Two-Branch Model: Chart + Numerical")
    loaders = create_dataloaders(config)
    chart = loaders['chart']
    num = loaders['numerical']

    feature_size = 64 if config['model']['chart_model'] == 'simplecnn' else 256
    chart_branch = get_chart_model(config['model']['chart_model'], 3, None)

    input_dim = next(iter(num['train']))[0].shape[1]
    num_branch = get_numerical_model(config, input_dim, feature_size)

    fusion = FusionModule(config['model']['fusion'], feature_size, num_branches=2)
    model = TwoBranchModel(chart_branch, num_branch, fusion).to(device)

    return train_multimodal_model(model, [chart['train']], num['train'], [chart['test']], num['test'], config)

# -----------------------
# Multi-chart
# -----------------------

def train_multi_chart_model(config):
    print("Training a Multi-Modal Chart-Only Model")
    loaders = create_dataloaders(config)
    train_loaders = loaders['train']
    test_loaders = loaders['test']

    chart_model = config['model']['chart_model']
    branches = [get_chart_model(chart_model, 3, None) for _ in train_loaders]

    fusion = FusionModule(config['model']['fusion'])
    model = MultiChartModel(branches, fusion).to(device)

    return train_multimodal_model(model, train_loaders, None, test_loaders, None, config)

# -----------------------
# Multi-chart + numerical
# -----------------------

def train_multi_chart_numerical_model(config):
    print("Training a Multi-Modal Chart + Numerical Model")
    loaders = create_dataloaders(config)
    chart = loaders['chart']
    num = loaders['numerical']

    chart_model = config['model']['chart_model']
    feature_size = 64 if chart_model == 'simplecnn' else 256
    branches = [get_chart_model(chart_model, 3, None) for _ in chart['train']]

    input_dim = next(iter(num['train']))[0].shape[1]
    num_branch = get_numerical_model(config, input_dim, feature_size)

    fusion = FusionModule(config['model']['fusion'], feature_size, num_branches=len(branches) + 1)
    model = MultiChartNumericalModel(branches, num_branch, fusion).to(device)

    return train_multimodal_model(model, chart['train'], num['train'], chart['test'], num['test'], config)

# -----------------------
# Numerical branch factory
# -----------------------

def get_numerical_model(config, input_dim, feature_size):
    branch_type = config['model']['numerical_branch']
    if branch_type == 'fcn':
        return NumericalFCN(input_dim=input_dim, output_dim=feature_size)
    elif branch_type == 'transformer':
        cfg = config.get('transformer_config', {})
        return NumericalTransformer(
            input_dim=input_dim,
            hidden_dim=cfg.get('hidden_dim', 128),
            num_heads=cfg.get('num_heads', 4),
            num_layers=cfg.get('num_layers', 2),
            dropout=cfg.get('dropout', 0.1),
            output_dim=cfg.get('output_dim', feature_size)
        )
    elif branch_type == 'oscnn':
        return NumericalOSCNN(output_dim=feature_size)
    else:
        raise ValueError(f"Unsupported numerical model: {branch_type}")

# -----------------------
# Standard single-modal trainer
# -----------------------

def train_standard_model(model, train_loader, test_loader, config):
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    patience, trigger_times, best_val_acc = 10, 0, 0

    for epoch in range(config['training']['epochs']):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Acc: {100*correct/total:.2f}%")
        scheduler.step(running_loss)

        if correct > best_val_acc:
            best_val_acc = correct
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping.")
                break

    return model

# -----------------------
# Multimodal trainer
# -----------------------

def train_multimodal_model(model, train_chart_loaders, train_num_loader, test_chart_loaders, test_num_loader, config):
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    patience, trigger_times, best_val_acc = 10, 0, 0

    def move(x): return [i.to(device) if isinstance(i, torch.Tensor) else i for i in x]

    for epoch in range(config['training']['epochs']):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        iterator = zip(zip(*train_chart_loaders), train_num_loader) if train_num_loader else zip(*train_chart_loaders)

        for batch in iterator:
            if train_num_loader:
                chart_batches, num_input = batch
                chart_batches = move(chart_batches)
                num_features, labels = move(num_input)
                chart_imgs = [img for img, _ in chart_batches]
                labels = chart_batches[0][1]
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

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_chart_loaders[0]):.4f}, Acc: {100*correct/total:.2f}%")
        scheduler.step(running_loss)

        if correct > best_val_acc:
            best_val_acc = correct
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping.")
                break

    return model
