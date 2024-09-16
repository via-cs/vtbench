import numpy as np
import torch
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV, train_test_split
from scipy.stats import randint


def read_ucr(filename):
    data = []
    labels = []
    
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) < 2:  # Ensure there's at least one feature and one label
                continue
            features = [float(f) for f in parts[:-1]]
            label = int(parts[-1].split(':')[-1])  # Handle label after the colon
            data.append(features)
            labels.append(label)
    
    print(f"Loaded {len(data)} samples from {filename}")
    return np.array(data), np.array(labels)

train_file = 'data/ECG5000/ECG5000_TRAIN.ts'
test_file = 'data/ECG5000/ECG5000_TEST.ts'
x_train, y_train = read_ucr(train_file)
x_test, y_test = read_ucr(test_file)

unique_labels = np.unique(y_train)
label_map = {label: idx for idx, label in enumerate(unique_labels)}
y_train = np.array([label_map[label] for label in y_train])
y_test = np.array([label_map[label] for label in y_test])

nb_classes = len(unique_labels)

print(f"Number of classes: {nb_classes}")
print(f"y_train unique labels: {np.unique(y_train)}")
print(f"y_test unique labels: {np.unique(y_test)}")

assert y_train.min() >= 0 and y_train.max() < nb_classes, "Train labels are out of range"
assert y_test.min() >= 0 and y_test.max() < nb_classes, "Test labels are out of range"

# Print shapes to ensure they match
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")

smote = SMOTE(k_neighbors=1)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

print(f"x_train_resampled shape: {x_train_resampled.shape}")
print(f"y_train_resampled shape: {y_train_resampled.shape}")
print(f"x_test reduced shape: {x_test.shape}")
print(f"y_test reduced shape: {y_test.shape}")

plt.hist(y_train_resampled, bins=nb_classes, edgecolor='k')
plt.title('Class Distribution in Training Set after SMOTE')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.xticks(ticks=np.arange(nb_classes))
plt.show()

X_train, X_val, y_train_resampled, y_val = train_test_split(
    x_train_resampled, y_train_resampled, test_size=0.2, random_state=42, stratify=y_train_resampled
)

param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': [None, 'balanced']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_model = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring='balanced_accuracy',
    cv=cv,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train_resampled)
best_params = grid_search.best_params_
print(f"Best hyperparameters: {best_params}")

# Best cross-validation score
best_cv_score = grid_search.best_score_
print(f"Best cross-validation balanced accuracy: {best_cv_score:.4f}")

best_model = grid_search.best_estimator_

# Evaluate on validation set
y_val_pred = best_model.predict(X_val)
val_balanced_acc = balanced_accuracy_score(y_val, y_val_pred)
print(f'Validation Balanced Accuracy with Best Model: {val_balanced_acc:.4f}')

# Evaluate on test set
y_test_pred = best_model.predict(x_test)
test_balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
print(f'Test Balanced Accuracy with Best Model: {test_balanced_acc:.4f}')

# Print results
print(f'Random Forest Validation Balanced Accuracy: {val_balanced_acc:.4f}')
print(f'Random Forest Test Balanced Accuracy: {test_balanced_acc:.4f}')
