import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from data.chart_generator import TimeSeriesImageDataset, NumericalDataset
from collections import Counter

def read_ucr(path):
    """
    Reads a UCR/UEA .ts file into (X, y).
    Handles headers, metadata, @data section, and colon/comma label formats.
    """
    X, y = [], []
    is_data = False
    has_label = False

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("@data"):
                is_data = True
                continue
            if not is_data:
                if line.lower().startswith("@classlabel") and "true" in line.lower():
                    has_label = True
                continue

            # --- data lines ---
            if ":" in line:
                try:
                    data_part, label_part = line.split(":")
                    values = [float(v) for v in data_part.replace(",", " ").split()]
                    label = int(float(label_part.strip()))
                    X.append(values)
                    y.append(label)
                    continue
                except Exception:
                    continue

            parts = line.replace("\t", " ").replace("  ", " ").split(",")
            if len(parts) == 1:
                parts = line.split()

            try:
                if has_label:
                    try:
                        label = int(float(parts[0]))
                        values = [float(v) for v in parts[1:]]
                    except ValueError:
                        values = [float(v) for v in parts[:-1]]
                        label = int(float(parts[-1]))
                    X.append(values)
                    y.append(label)
                else:
                    X.append([float(v) for v in parts])
            except Exception:
                continue

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int) if has_label else np.zeros(len(X), dtype=int)

    # Normalize labels to start from 0
    if len(np.unique(y)) > 0:
        unique_labels = np.unique(y)
        label_map = {old: i for i, old in enumerate(unique_labels)}
        y = np.array([label_map[val] for val in y], dtype=int)

    print(f" Loaded {path}: {X.shape[0]} samples, {X.shape[1] if X.ndim>1 else 0} features, {len(np.unique(y))} classes")
    return X, y


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
        if model_type in ['two_branch', 'multi_modal_chart'] and config['model'].get('numerical_branch', 'none') != 'none':
            numerical_loader = DataLoader(numerical_datasets[split], batch_size=batch_size, shuffle=shuffle)

        if model_type == 'single_modal_chart':
            chart_loaders = chart_loaders[0]

        dataloaders[split] = {
            'chart': chart_loaders,
            'numerical': numerical_loader
        }

    return dataloaders
