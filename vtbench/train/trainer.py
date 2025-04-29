import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from vtbench.data.loader import read_ucr, create_dataloaders, NumericalDataset
from vtbench.models.cnn.simplecnn import SimpleCNN
from vtbench.models.cnn.deepcnn import DeepCNN
from vtbench.models.numerical.fcn import NumericalFCN
from vtbench.models.numerical.transformer import NumericalTransformer
from vtbench.models.numerical.oscnn import NumericalOSCNN
from vtbench.models.multimodal.two_branch import TwoBranchModel
from vtbench.models.multimodal.multi_chart import MultiChartModel
from vtbench.models.multimodal.multi_chart_numerical import MultiChartNumericalModel
from vtbench.models.multimodal.fusion import FusionModule
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)



# ========================
# Controller: Main trainer
# ========================

def train_model(config):
    model_type = config['model']['type']

    if model_type == 'single_modal_chart':
        model = train_single_chart_model(config)

    elif model_type == 'two_branch':
        model = train_two_branch_model(config)

    elif model_type == 'multi_modal_chart':
        model = train_multi_chart_model(config)

    elif model_type == 'multi_modal_chart_numerical':
        model = train_multi_chart_numerical_model(config)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model

# -----------------------
# Train single chart model
# -----------------------

def train_single_chart_model(config):
    """
    Train a single chart model (SimpleCNN / DeepCNN) using one branch from chart_branches.
    """

    print(f"Training a single chart model: {config['model']['cnn_arch']}")

    train_datasets = create_dataloaders(config, split='train')
    test_datasets = create_dataloaders(config, split='test')
    
    train_loader = DataLoader(train_datasets[0], batch_size=config['training']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_datasets[0], batch_size=config['training']['batch_size'], shuffle=False)

    cnn_arch = config['model']['cnn_arch']
    if cnn_arch == 'simplecnn':
        model = SimpleCNN(input_channels = 3, num_classes = None)
    elif cnn_arch == 'deepcnn':
        model = DeepCNN(input_channels = 3, num_classes = None)
    else:
        raise ValueError(f"Unsupported CNN architecture: {cnn_arch}")

    model = model.to(device)

    model = train_standard_model(model, train_loader, test_loader, config)

    return model


# -----------------------
# Train 2 branch model
# -----------------------


def train_two_branch_model(config):
    """
    Train a 2-branch multimodal model:
    - One chart branch
    - One numerical branch
    """

    print(f"Training a Two-Branch Model: Chart + Numerical")

    # Load chart image dataset
    train_datasets = create_dataloaders(config, split='train')
    test_datasets = create_dataloaders(config, split='test')

    train_loader = DataLoader(train_datasets[0], batch_size=config['training']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_datasets[0], batch_size=config['training']['batch_size'], shuffle=False)

    # Load numerical dataset
    train_num_data, train_num_labels = read_ucr(config['dataset']['train_path'])
    test_num_data, test_num_labels = read_ucr(config['dataset']['test_path'])

    train_num_loader = DataLoader(NumericalDataset(train_num_data, train_num_labels), batch_size=config['training']['batch_size'], shuffle=True)
    test_num_loader = DataLoader(NumericalDataset(test_num_data, test_num_labels), batch_size=config['training']['batch_size'], shuffle=False)

    # Instantiate chart model
    cnn_arch = config['model']['cnn_arch']
    feature_size = 64 if cnn_arch == 'simplecnn' else 256

    if cnn_arch == 'simplecnn':
        chart_branch = SimpleCNN(input_channels = 3, num_classes = None)
    elif cnn_arch == 'deepcnn':
        chart_branch = DeepCNN(input_channels = 3, num_classes = None)
    else:
        raise ValueError(f"Unsupported CNN architecture: {cnn_arch}")

    # Instantiate numerical model
    input_dim = len(train_num_data[0])
    numerical_branch = config['model']['numerical_branch']

    if numerical_branch == 'fcn':
        num_branch = NumericalFCN(input_dim=input_dim, output_dim=feature_size)

    elif numerical_branch == 'transformer':
        transformer_cfg = {
            'hidden_dim': config.get('transformer_hidden_dim', 128),
            'num_heads': config.get('transformer_heads', 4),
            'num_layers': config.get('transformer_layers', 2),
            'dropout': config.get('transformer_dropout', 0.1),
            'output_dim': feature_size
        }
        num_branch = NumericalTransformer(
            input_dim=input_dim,
            hidden_dim=transformer_cfg['hidden_dim'],
            num_heads=transformer_cfg['num_heads'],
            num_layers=transformer_cfg['num_layers'],
            dropout=transformer_cfg['dropout'],
            output_dim=transformer_cfg['output_dim']
        )

    elif numerical_branch == 'oscnn':
        num_branch = NumericalOSCNN(output_dim=feature_size)

    else:
        raise ValueError(f"Unsupported numerical branch: {numerical_branch}")

    # Fusion module
    fusion = FusionModule(
        mode=config['model']['fusion'],
        feature_size=feature_size,
        num_branches=2
    )

    model = TwoBranchModel(chart_branch, num_branch, fusion)
    model = model.to(device)

    # Train
    model = train_multimodal_model(
        model,
        [train_loader],
        train_num_loader,
        [test_loader],
        test_num_loader,
        config
    )

    return model


# -----------------------
# Train multi-chart model
# -----------------------

def train_multi_chart_model(config):
    """
    Train a multi-chart model:
    - Multiple chart branches
    - No numerical branch
    """

    print(f"Training a Multi-Modal Chart-Only Model")

    train_datasets = create_dataloaders(config, split='train')
    test_datasets = create_dataloaders(config, split='test')

    train_loaders = [DataLoader(ds, batch_size=config['training']['batch_size'], shuffle=True) for ds in train_datasets]
    test_loaders = [DataLoader(ds, batch_size=config['training']['batch_size'], shuffle=False) for ds in test_datasets]

    cnn_arch = config['model']['cnn_arch']
    branches = []

    for _ in train_loaders:
        if cnn_arch == 'simplecnn':
            branches.append(SimpleCNN(input_channels = 3, num_classes = None))
        elif cnn_arch == 'deepcnn':
            branches.append(DeepCNN(input_channels = 3, num_classes = None))
        else:
            raise ValueError(f"Unsupported CNN architecture: {cnn_arch}")

    fusion = FusionModule(mode=config['model']['fusion'])

    model = MultiChartModel(branches, fusion)
    model = model.to(device)

    model = train_multimodal_model(model, train_loaders, None, test_loaders, None, config)

    return model

# -----------------------
# Train multi-chart + numerical model
# -----------------------

def train_multi_chart_numerical_model(config):
    """
    Train a multi-chart + numerical model:
    - Multiple chart branches
    - Numerical branch (FCN, Transformer, OSCNN)
    """

    print(f"Training a Multi-Modal Chart + Numerical Model")

    train_datasets = create_dataloaders(config, split='train')
    test_datasets = create_dataloaders(config, split='test')

    train_loaders = [DataLoader(ds, batch_size=config['training']['batch_size'], shuffle=True) for ds in train_datasets]
    test_loaders = [DataLoader(ds, batch_size=config['training']['batch_size'], shuffle=False) for ds in test_datasets]

    # Load numerical data
    train_num_data, train_num_labels = read_ucr(config['dataset']['train_path'])
    test_num_data, test_num_labels = read_ucr(config['dataset']['test_path'])

    train_num_loader = DataLoader(NumericalDataset(train_num_data, train_num_labels), batch_size=config['training']['batch_size'], shuffle=True)
    test_num_loader = DataLoader(NumericalDataset(test_num_data, test_num_labels), batch_size=config['training']['batch_size'], shuffle=False)

    # Build chart branches
    cnn_arch = config['model']['cnn_arch']
    branches = []
    for _ in train_loaders:
        if cnn_arch == 'simplecnn':
            branches.append(SimpleCNN(input_channels = 3, num_classes = None))
        elif cnn_arch == 'deepcnn':
            branches.append(DeepCNN(input_channels = 3, num_classes = None))
        else:
            raise ValueError(f"Unsupported CNN architecture: {cnn_arch}")

    # Build numerical branch
    numerical_branch = config['model']['numerical_branch']
    feature_size = 64 if config['model']['cnn_arch'] == 'simplecnn' else 256
    input_dim = len(train_num_data[0])

    if numerical_branch == 'fcn':
        num_branch = NumericalFCN(input_dim=input_dim, output_dim=feature_size)

    elif numerical_branch == 'transformer':
        transformer_cfg = {
            'hidden_dim': config.get('transformer_hidden_dim', 128),
            'num_heads': config.get('transformer_heads', 4),
            'num_layers': config.get('transformer_layers', 2),
            'dropout': config.get('transformer_dropout', 0.1),
            'output_dim': feature_size
        }
        num_branch = NumericalTransformer(
            input_dim=input_dim,
            hidden_dim=transformer_cfg['hidden_dim'],
            num_heads=transformer_cfg['num_heads'],
            num_layers=transformer_cfg['num_layers'],
            dropout=transformer_cfg['dropout'],
            output_dim=transformer_cfg['output_dim']
        )

    elif numerical_branch == 'oscnn':
        num_branch = NumericalOSCNN(output_dim=feature_size)

    else:
        raise ValueError(f"Unsupported numerical branch: {numerical_branch}")


    num_branches = len(train_loaders) + (1 if numerical_branch != 'none' else 0)
    fusion = FusionModule(
        mode=config['model']['fusion'],
        feature_size=feature_size,
        num_branches=num_branches
    )

    model = MultiChartNumericalModel(branches, num_branch, fusion)
    model = model.to(device)

    model = train_multimodal_model(model, train_loaders, train_num_loader, test_loaders, test_num_loader, config)

    return model

# -----------------------
# Standard train function for single model
# -----------------------

def train_standard_model(model, train_loader, test_loader, config):
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    patience = 10
    trigger_times = 0
    best_val_acc = 0

    for epoch in range(config['training']['epochs']):
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
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}], Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.2f}%")

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.inference_mode():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(test_loader)
        val_acc = 100 * correct / total
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

    return model

# -----------------------
# Train multimodal model (multi inputs)
# -----------------------

def train_multimodal_model(model, train_chart_loader, train_num_loader, test_chart_loader, test_num_loader, config):
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    patience = 10
    trigger_times = 0
    best_val_acc = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def move_to_device(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, (list, tuple)):
            return [move_to_device(i) for i in x]
        else:
            return x

    for epoch in range(config['training']['epochs']):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for chart_batches, num_batches in zip(zip(*train_chart_loader), train_num_loader):
            chart_inputs, num_input = chart_batches, num_batches

            chart_inputs = move_to_device(chart_inputs)
            num_features, labels = move_to_device(num_input)

            labels = labels.to(device)

            optimizer.zero_grad()
            chart_imgs = [img for img, _ in chart_inputs]  # get only images, not labels
            outputs = model((chart_imgs, num_features))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_chart_loader[0])
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}], Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.2f}%")

        scheduler.step(epoch_loss)

        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

    return model

