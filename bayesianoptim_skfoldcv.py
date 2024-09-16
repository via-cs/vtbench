import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from collections import Counter
from imblearn.over_sampling import SMOTE
from vtbench.data_utils import read_ucr, normalize_data, apply_smote, to_torch_tensors
import optuna
from optuna.samplers import TPESampler 
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
    plot_intermediate_values
)

from vtbench.models.SimpleCNN import Simple2DCNN
from vtbench.models.DeepCNN import Deep2DCNN
from vtbench.CNN_train import create_dataloaders, train_model, evaluate_model_cv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_file = 'data/ECG5000/ECG5000_TRAIN.ts'
test_file = 'data/ECG5000/ECG5000_TEST.ts'

global x_train, y_train, x_test, y_test
x_train, y_train = read_ucr(train_file)
x_test, y_test = read_ucr(test_file)

print(f"Initial x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"Initial x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

unique_labels = np.unique(y_train)
label_map = {label: idx for idx, label in enumerate(unique_labels)}
y_train = np.array([label_map[label] for label in y_train])
y_test = np.array([label_map[label] for label in y_test])

nb_classes = len(unique_labels)

x_train, x_test = normalize_data(x_train, x_test)

print(f"After normalization x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")


if x_train is None or y_train is None or x_test is None or y_test is None:
    raise ValueError("Training or testing data is not properly initialized. Please check your data loading.")

excel_file_path = 'bayesian_optim_skfoldcv_results.xlsx'

def write_results_to_excel(file_path, results, version):
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        df = pd.DataFrame()

    results_df = pd.DataFrame([results])

    results_df['Version'] = version

    df = pd.concat([df, results_df], ignore_index=True)

    with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
        df.to_excel(writer, index=False)

def objective(trial):
    global x_train, y_train, x_test, y_test

    x_train, y_train = read_ucr(train_file)
    x_test, y_test = read_ucr(test_file)

    unique_labels = np.unique(y_train)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y_train = np.array([label_map[label] for label in y_train])
    y_test = np.array([label_map[label] for label in y_test])

    nb_classes = len(unique_labels)
    x_train, x_test = normalize_data(x_train, x_test)

    lr = trial.suggest_float('lr', 1e-5, 2e-4, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-3, 1.5e-3, log=True)
    model_name = trial.suggest_categorical('model', ['Simple2DCNN', 'Deep2DCNN'])
    patience = trial.suggest_int('patience', 7, 10)
    apply_smote_flag = trial.suggest_categorical('apply_smote', [True, False])
    combo_key = trial.suggest_categorical('combo_key', list(create_dataloaders(x_train, y_train, x_test, y_test).keys()))
    batch_size = trial.suggest_categorical('batch_size', [16, 32])
    optimizer_type = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
    dropout_rate = trial.suggest_float('dropout_rate', 0.45, 0.5)
    kernel_size = trial.suggest_categorical('kernel_size', [2, 7])

    print(f"Running combination: Model={model_name}, Apply SMOTE={apply_smote_flag}, Combination={combo_key}, "
          f"Batch Size={batch_size}, Optimizer={optimizer_type}, Dropout Rate={dropout_rate}, Kernel Size={kernel_size}, "
          f"Learning Rate={lr}, Weight Decay={weight_decay}, Patience={patience}")

    skf = StratifiedKFold(n_splits=5, shuffle = True, random_state=42)
    
    accuracies = []
    validation_accuracies = []
    validation_losses = []

    for fold, (train_index, val_index) in enumerate(skf.split(x_train, y_train), 1):
        print(f"Fold {fold}")
    
        X_train_fold, X_val_fold = x_train[train_index], x_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
        min_class_samples = min(Counter(y_train_fold).values())
        
        if apply_smote_flag and min_class_samples > 1:  
            smote_k_neighbors = max(min(min_class_samples - 1, 5), 1) 
            smote = SMOTE(k_neighbors=smote_k_neighbors)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_fold, y_train_fold)
            print(f"Applied SMOTE on fold {fold} with k_neighbors={smote_k_neighbors}")
        else:
            X_train_resampled, y_train_resampled = X_train_fold, y_train_fold
            print(f"Skipped SMOTE on fold {fold} due to insufficient samples in the smallest class")

    
        placeholder_x_test = np.empty((0, X_train_fold.shape[1]))
        placeholder_y_test = np.empty((0,))
        X_train_fold, y_train_fold, _, _ = to_torch_tensors(X_train_resampled, y_train_resampled, placeholder_x_test, placeholder_y_test)
        X_val_fold, y_val_fold, _, _ = to_torch_tensors(X_val_fold, y_val_fold, placeholder_x_test, placeholder_y_test)
                
        model_class = Simple2DCNN if model_name == 'Simple2DCNN' else Deep2DCNN
        model = model_class(3, nb_classes).to(device)

        dataloaders = create_dataloaders(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
        train_loader, val_loader, _ = dataloaders[combo_key]

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) if optimizer_type == 'Adam' else optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

        num_epochs = 100
        best_val_loss, best_val_accuracy = train_model(
            model,
            train_loader,
            val_loader,
            num_epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            patience=patience
        )

        
        metrics = evaluate_model_cv(model, val_loader)
        fold_balanced_accuracy = metrics['balanced_accuracy']
        fold_val_accuracy = metrics['val_accuracy']  # Assuming 'val_accuracy' is returned in your metrics
        fold_val_loss = metrics['val_loss'] 

        accuracies.append(fold_balanced_accuracy)
        validation_accuracies.append(fold_val_accuracy)
        validation_losses.append(fold_val_loss)

        print(f"Fold {fold}: Balanced Accuracy = {fold_balanced_accuracy:.4f}")
        
    avg_balanced_accuracy = np.mean(accuracies)
    avg_validation_accuracy = np.mean(validation_accuracies)
    avg_validation_loss = np.mean(validation_losses)

    print(f"Average Balanced Accuracy across all folds: {avg_balanced_accuracy:.4f}")
    print(f"Average Validation Accuracy across all folds: {avg_validation_accuracy:.4f}")
    print(f"Average Validation Loss across all folds: {avg_validation_loss:.4f}")

    results = {
        'SMOTE': apply_smote_flag,
        'Model': model_name,
        'Combination': combo_key,
        'Average Balanced Accuracy': avg_balanced_accuracy,
        'Average Validation Loss': avg_validation_loss,
        'Average Validation Accuracy': avg_validation_accuracy,
        'Learning Rate': lr,
        'Weight Decay': weight_decay,
        'Patience': patience,
        'Batch Size': batch_size,
        'Optimizer': optimizer_type,
        'Dropout Rate': dropout_rate,
        'Kernel Size': kernel_size
    }

    write_results_to_excel(excel_file_path, results, "Bayesian Optimization")

    return avg_balanced_accuracy

sampler = TPESampler()
study = optuna.create_study(direction='maximize', sampler=sampler)

study.optimize(objective, n_trials=200)  

fig = plot_optimization_history(study)
fig.show()

fig = plot_param_importances(study)
fig.show()

fig = plot_parallel_coordinate(study)
fig.show()

fig = plot_slice(study)
fig.show()

fig = plot_intermediate_values(study)
fig.show()

best_hyperparameters = study.best_params
print("Best Hyperparameters: ", best_hyperparameters)
