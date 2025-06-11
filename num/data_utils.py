import numpy as np
from imblearn.over_sampling import SMOTE
import torch
from collections import Counter

import numpy as np
import torch
from collections import Counter

import numpy as np
from collections import Counter

def read_ucr(filename):
    data = []
    labels = []
    raw_labels = []

    # First pass to collect all raw class names
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            label_str = parts[-1].split(':')[-1].strip()
            raw_labels.append(label_str)

    # Create mapping based on alphabetical order
    unique_labels = sorted(set(raw_labels))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    normalize = lambda x: label_map[x]

    # Second pass to convert data and labels
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            features = [float(f) for f in parts[:-1]]
            label_str = parts[-1].split(':')[-1].strip()
            labels.append(normalize(label_str))
            data.append(features)

    data = np.array(data)
    labels = np.array(labels)

    print(f"{filename} loaded. Total samples: {len(labels)}, Class distribution: {Counter(labels)}")
    return data, labels, label_map



def read_ecg5000(filename):
    data = []
    labels = []
    
    def normalize(label):
        if label == 1:
            return 0  # Normal
        elif label in {2, 3, 4}:
            return 1  # Abnormal
        else:
            return None  

   
    with open(filename, 'r') as file:
        for line in file:
           
            parts = line.strip().split(':')
            features = list(map(float, parts[0].split(',')))
            label = int(parts[1])
           
            normalized_label = normalize(label)
            if normalized_label is not None:  
                data.append(features)
                labels.append(normalized_label)

    data = np.array(data)
    labels = np.array(labels)

    if len(labels) > 0:
        print(f"ECG5000 Data loaded. Total samples: {len(labels)}, Class distribution: {np.bincount(labels)}")
    else:
        print(f"Warning: No valid samples found in {filename}. Check the file format or filtering criteria.")

    return data, labels




def normalize_data(x_train, x_test):
    x_train_mean = x_train.mean()
    x_train_std = x_train.std()
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std
    return x_train, x_test

def to_torch_tensors(x_train, y_train, x_test, y_test):
    X_train = torch.tensor(x_train, dtype=torch.float32)
    X_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    return X_train, y_train, X_test, y_test






