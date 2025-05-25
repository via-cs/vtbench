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


def build_chart_datasets(X, y, split, dataset_name, chart_branches, transform, generate_images=False, overwrite_existing=False, global_indices=None):
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
            bar_mode=branch_cfg.get('bar_mode', 'border'),
            transform=transform,
            generate_images=generate_images,
            overwrite_existing=overwrite_existing,
            global_indices=global_indices if global_indices is not None else list(range(len(y)))
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

    transform_train = transforms.Compose(base_transforms)
    transform_eval = transforms.Compose(base_transforms)

    # Load raw data
    X_train, y_train = read_ucr(config['dataset']['train_path'])
    X_test, y_test = read_ucr(config['dataset']['test_path'])

    # Print label dist for sanity
    print("Train labels:", Counter(y_train))
    print("Test labels:", Counter(y_test))

    temp_ds = torch.utils.data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    val_ds, test_ds = stratified_val_test_split(temp_ds, y_test, val_size=0.2, seed=seed)

    val_indices = val_ds.indices
    test_indices = test_ds.indices

    # Build chart datasets
    chart_datasets = {
        'train': build_chart_datasets(X_train, y_train, 'train', dataset_name, chart_branches, transform_train,
                                      generate_images=config['image_generation'].get('generate_images', False),
                                      overwrite_existing=config['image_generation'].get('overwrite_existing', False)),

        'val': build_chart_datasets(X_test[val_indices], y_test[val_indices], 'test', dataset_name, chart_branches, transform_eval,
                                    generate_images=config['image_generation'].get('generate_images', False),
                                    overwrite_existing=config['image_generation'].get('overwrite_existing', False),
                                    global_indices=val_indices),

        'test': build_chart_datasets(X_test[test_indices], y_test[test_indices], 'test', dataset_name, chart_branches, transform_eval,
                                     generate_images=config['image_generation'].get('generate_images', False),
                                     overwrite_existing=config['image_generation'].get('overwrite_existing', False),
                                     global_indices=test_indices)
    }

    numerical_datasets = {
        'train': NumericalDataset(X_train, y_train),
        'val': NumericalDataset(X_test[val_indices], y_test[val_indices]),
        'test': NumericalDataset(X_test[test_indices], y_test[test_indices])
    }

    # Create final loaders
    dataloaders = {}
    for split in ['train', 'val', 'test']:
        shuffle = (split == 'train')
        chart_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=shuffle) for ds in chart_datasets[split]]

        numerical_loader = None
        if model_type in ['two_branch', 'multi_modal_chart_numerical'] and config['model'].get('numerical_branch', 'none') != 'none':
            numerical_loader = DataLoader(numerical_datasets[split], batch_size=batch_size, shuffle=shuffle)

        if model_type == 'single_modal_chart':
            chart_loaders = chart_loaders[0]

        dataloaders[split] = {
            'chart': chart_loaders,
            'numerical': numerical_loader
        }

    return dataloaders
