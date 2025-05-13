import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from vtbench.data.chart_generator import TimeSeriesImageDataset, NumericalDataset
from sklearn.model_selection import StratifiedShuffleSplit

# ------------------------
# Utility to load `.ts` files
# ------------------------

def read_ucr(filename):
    data = []
    labels = []
    label_set = set()  

    print(f"Loading file: {filename}")

    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            label = int(parts[-1].split(':')[-1])
            label_set.add(label)

 
    if label_set == {0, 1}: 
        def normalize(label):
            return label
    elif label_set == {1, 2}:  
        def normalize(label):
            return 0 if label == 1 else 1
    elif label_set == {-1, 1}:  
        def normalize(label):
            return 0 if label == -1 else 1
    else:
        raise ValueError(f"Unexpected label set: {label_set}")

    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            features = [float(f) for f in parts[:-1]]
            label = int(parts[-1].split(':')[-1])
            normalized_label = normalize(label)
            labels.append(normalized_label)
            data.append(features)

    print(f"Finished loading {filename} - Samples: {len(labels)}")
    return np.array(data), np.array(labels)


def create_dataloaders(config, seed=42):
    """
    Create train, val, test dataloaders with consistent splits and label alignment.
    Validation is created from the test set (20% of test).
    """

    model_type = config['model']['type']
    chart_branches = config['chart_branches']
    dataset_name = config['dataset']['name']
    batch_size = config['training']['batch_size']

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # === Load .ts files ===
    X_train, y_train = read_ucr(config['dataset']['train_path'])
    X_test_full, y_test_full = read_ucr(config['dataset']['test_path'])

    # === Stratified split: test → val + test_final ===
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.8, random_state=seed)
    val_idx, test_idx = next(sss.split(X_test_full, y_test_full))
    X_val, y_val = X_test_full[val_idx], y_test_full[val_idx]
    X_test, y_test = X_test_full[test_idx], y_test_full[test_idx]

    # === Chart dataset builder ===
    def make_chart_dataset(X, y, split_type):
        datasets = []
        for branch_cfg in chart_branches.values():
            ds = TimeSeriesImageDataset(
                dataset_name=dataset_name,
                time_series_data=X,
                labels=y,
                split=split_type,
                chart_type=branch_cfg['chart_type'],
                color_mode=branch_cfg.get('color_mode', 'color'),
                label_mode=branch_cfg.get('label_mode', 'with_label'),
                scatter_mode=branch_cfg.get('scatter_mode', 'plain'),
                bar_mode=branch_cfg.get('bar_mode', 'fill'),
                transform=transform
            )
            datasets.append(ds)
        return datasets

    # === Handle model types ===

    if model_type == 'single_modal_chart':
        first_branch_cfg = list(chart_branches.values())[0]

        def build_single_chart_loader(X, y, split_type):
            dataset = TimeSeriesImageDataset(
                dataset_name=dataset_name,
                time_series_data=X,
                labels=y,
                split=split_type,
                chart_type=first_branch_cfg['chart_type'],
                color_mode=first_branch_cfg.get('color_mode', 'color'),
                label_mode=first_branch_cfg.get('label_mode', 'with_label'),
                scatter_mode=first_branch_cfg.get('scatter_mode', 'plain'),
                bar_mode=first_branch_cfg.get('bar_mode', 'fill'),
                transform=transform
            )
            return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        return {
            'train': build_single_chart_loader(X_train, y_train, 'train'),
            'val': build_single_chart_loader(X_val, y_val, 'val'),
            'test': build_single_chart_loader(X_test, y_test, 'test'),
        }

    elif model_type == 'multi_modal_chart':
        return {
            'train': [DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)
                      for ds in make_chart_dataset(X_train, y_train, 'train')],
            'val': [DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)
                    for ds in make_chart_dataset(X_val, y_val, 'val')],
            'test': [DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)
                     for ds in make_chart_dataset(X_test, y_test, 'test')],
        }

    elif model_type == 'multi_modal_chart_numerical':
        return {
            'chart': {
                'train': [DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)
                          for ds in make_chart_dataset(X_train, y_train, 'train')],
                'val': [DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)
                        for ds in make_chart_dataset(X_val, y_val, 'val')],
                'test': [DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)
                         for ds in make_chart_dataset(X_test, y_test, 'test')],
            },
            'numerical': {
                'train': DataLoader(NumericalDataset(X_train, y_train), batch_size=batch_size, shuffle=False, drop_last=True),
                'val': DataLoader(NumericalDataset(X_val, y_val), batch_size=batch_size, shuffle=False, drop_last=True),
                'test': DataLoader(NumericalDataset(X_test, y_test), batch_size=batch_size, shuffle=False, drop_last=True),
            }
        }

    elif model_type == 'two_branch':
        first_branch_cfg = list(chart_branches.values())[0]
        chart_train = TimeSeriesImageDataset(
            dataset_name=dataset_name,
            time_series_data=X_train,
            labels=y_train,
            split='train',
            chart_type=first_branch_cfg['chart_type'],
            color_mode=first_branch_cfg.get('color_mode', 'color'),
            label_mode=first_branch_cfg.get('label_mode', 'with_label'),
            scatter_mode=first_branch_cfg.get('scatter_mode', 'plain'),
            bar_mode=first_branch_cfg.get('bar_mode', 'fill'),
            transform=transform
        )
        chart_val = TimeSeriesImageDataset(
            dataset_name=dataset_name,
            time_series_data=X_val,
            labels=y_val,
            split='val',
            chart_type=first_branch_cfg['chart_type'],
            color_mode=first_branch_cfg.get('color_mode', 'color'),
            label_mode=first_branch_cfg.get('label_mode', 'with_label'),
            scatter_mode=first_branch_cfg.get('scatter_mode', 'plain'),
            bar_mode=first_branch_cfg.get('bar_mode', 'fill'),
            transform=transform
        )
        chart_test = TimeSeriesImageDataset(
            dataset_name=dataset_name,
            time_series_data=X_test,
            labels=y_test,
            split='test',
            chart_type=first_branch_cfg['chart_type'],
            color_mode=first_branch_cfg.get('color_mode', 'color'),
            label_mode=first_branch_cfg.get('label_mode', 'with_label'),
            scatter_mode=first_branch_cfg.get('scatter_mode', 'plain'),
            bar_mode=first_branch_cfg.get('bar_mode', 'fill'),
            transform=transform
        )

        return {
            'chart': {
                'train': DataLoader(chart_train, batch_size=batch_size, shuffle=False, drop_last=True),
                'val': DataLoader(chart_val, batch_size=batch_size, shuffle=False, drop_last=True),
                'test': DataLoader(chart_test, batch_size=batch_size, shuffle=False, drop_last=True),
            },
            'numerical': {
                'train': DataLoader(NumericalDataset(X_train, y_train), batch_size=batch_size, shuffle=False, drop_last=True),
                'val': DataLoader(NumericalDataset(X_val, y_val), batch_size=batch_size, shuffle=False, drop_last=True),
                'test': DataLoader(NumericalDataset(X_test, y_test), batch_size=batch_size, shuffle=False, drop_last=True),
            }
        }

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


