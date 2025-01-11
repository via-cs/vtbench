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
            label = int(parts[-1].split(':')[-1]) - 1
            data.append(features)
            labels.append(label)
    return np.array(data), np.array(labels)

# def load_test_data(config):
#     """Load and preprocess the ECG5000 dataset."""
#     train_file = 'data/ECG5000/ECG5000_TRAIN.ts'
#     test_file = 'data/ECG5000/ECG5000_TEST.ts'

#     x_train, y_train = read_ucr(train_file)
#     x_test, y_test = read_ucr(test_file)

#     x_train, x_test = normalize_data(x_train, x_test)


#     unique_labels = np.unique(np.concatenate([y_train, y_test]))
#     label_map = {label: idx for idx, label in enumerate(unique_labels)}
#     y_train = np.array([label_map[label] for label in y_train])
#     y_test = np.array([label_map[label] for label in y_test])

#     X_train, y_train, X_test, y_test = to_torch_tensors(x_train, y_train, x_test, y_test)

#     print(f"Classes in Training Set: {np.unique(y_train)}")
#     print(f"Classes in Test Set: {np.unique(y_test)}")

#     return X_train, y_train, X_test, y_test

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
