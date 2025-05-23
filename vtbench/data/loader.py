import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from vtbench.data.chart_generator import TimeSeriesImageDataset, NumericalDataset
from collections import Counter

def read_ucr(filename):
    data, labels = [], []
    label_set = set()
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            label = int(parts[-1].split(':')[-1])
            label_set.add(label)

    if label_set == {0, 1}:
        normalize = lambda l: l
    elif label_set == {1, 2}:
        normalize = lambda l: 0 if l == 1 else 1
    elif label_set == {-1, 1}:
        normalize = lambda l: 0 if l == -1 else 1
    else:
        raise ValueError(f"Unexpected label set: {label_set}")

    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            features = [float(f) for f in parts[:-1]]
            label = int(parts[-1].split(':')[-1])
            data.append(features)
            labels.append(normalize(label))

    return np.array(data), np.array(labels)


def stratified_val_test_split(dataset, labels, val_size=0.2, seed=42):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    indices = np.arange(len(dataset))
    for val_idx, test_idx in sss.split(indices, labels):
        return Subset(dataset, val_idx), Subset(dataset, test_idx)


def build_chart_datasets(X, y, split, dataset_name, chart_branches, transform):
    datasets = []
    for branch_cfg in chart_branches.values():
        ds = TimeSeriesImageDataset(
            dataset_name=dataset_name,
            time_series_data=X,
            labels=y,
            split=split,
            chart_type=branch_cfg['chart_type'],
            color_mode=branch_cfg.get('color_mode', 'color'),
            label_mode=branch_cfg.get('label_mode', 'with_label'),
            scatter_mode=branch_cfg.get('scatter_mode', 'plain'),
            bar_mode=branch_cfg.get('bar_mode', 'fill'),
            transform=transform,
        )
        datasets.append(ds)
    return datasets


def create_dataloaders(config, seed=42):
    model_type = config['model']['type']
    chart_branches = config.get('chart_branches', {})
    dataset_name = config['dataset']['name']
    batch_size = config['training']['batch_size']

    base_transforms = [
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ]
    aug_transforms = [
        transforms.RandomRotation(degrees=5),
        transforms.RandomResizedCrop(size=(64, 64), scale=(0.9, 1.0)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
    ]

    transform_train = transforms.Compose(base_transforms) 
    transform_eval = transforms.Compose(base_transforms)

    # Load raw data
    X_train, y_train = read_ucr(config['dataset']['train_path'])
    X_test, y_test = read_ucr(config['dataset']['test_path'])

    # Print label dist for sanity
    print("Train labels:", Counter(y_train))
    print("Test labels:", Counter(y_test))

    # Build test dataset to split into val + test subsets
    temp_ds = TimeSeriesImageDataset(
        dataset_name=dataset_name,
        time_series_data=X_test,
        labels=y_test,
        split='test',
        chart_type=list(chart_branches.values())[0]['chart_type'],
        color_mode=list(chart_branches.values())[0].get('color_mode', 'color'),
        label_mode=list(chart_branches.values())[0].get('label_mode', 'with_label'),
        transform=transform_eval
    )
    val_ds, test_ds = stratified_val_test_split(temp_ds, y_test, val_size=0.2, seed=seed)

    # Create final datasets for chart input
    chart_datasets = {
        'train': build_chart_datasets(X_train, y_train, 'train', dataset_name, chart_branches, transform_train),
        'val': [Subset(ds, val_ds.indices) for ds in build_chart_datasets(X_test, y_test, 'test', dataset_name, chart_branches, transform_eval)],
        'test': [Subset(ds, test_ds.indices) for ds in build_chart_datasets(X_test, y_test, 'test', dataset_name, chart_branches, transform_eval)]
    }

    # Numerical datasets
    numerical_datasets = {
        'train': NumericalDataset(X_train, y_train),
        'val': Subset(NumericalDataset(X_test, y_test), val_ds.indices),
        'test': Subset(NumericalDataset(X_test, y_test), test_ds.indices)
    }

    dataloaders = {}
    for split in ['train', 'val', 'test']:
        shuffle = (split == 'train')
        chart_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True) for ds in chart_datasets[split]]

        numerical_loader = None
        if model_type in ['two_branch', 'multi_modal_chart_numerical'] and config['model'].get('numerical_branch', 'none') != 'none':
            numerical_loader = DataLoader(numerical_datasets[split], batch_size=batch_size, shuffle=shuffle, drop_last=True)

        if model_type == 'single_modal_chart':
            chart_loaders = chart_loaders[0]

        dataloaders[split] = {
            'chart': chart_loaders,
            'numerical': numerical_loader
        }

    return dataloaders