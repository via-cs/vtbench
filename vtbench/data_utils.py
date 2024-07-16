import numpy as np
from imblearn.over_sampling import SMOTE
import torch

def read_ucr(filename):
    data = []
    labels = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            features = [float(f) for f in parts[:-1]]
            label = int(parts[-1].split(':')[-1])
            data.append(features)
            labels.append(label)
    print(f"Loaded {len(data)} samples from {filename}")
    return np.array(data), np.array(labels)

def normalize_data(x_train, x_test):
    x_train_mean = x_train.mean()
    x_train_std = x_train.std()
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std
    return x_train, x_test

def apply_smote(x_train, y_train, strategy):
    smote = SMOTE(sampling_strategy=strategy, k_neighbors=1)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
    return x_train_resampled, y_train_resampled

def to_torch_tensors(x_train, y_train, x_test, y_test):
    X_train = torch.tensor(x_train, dtype=torch.float32)
    X_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    return X_train, y_train, X_test, y_test
