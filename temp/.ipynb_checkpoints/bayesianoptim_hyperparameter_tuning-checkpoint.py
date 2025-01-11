import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from vtbench.data_utils import read_ucr, normalize_data, apply_smote, to_torch_tensors
import optuna
from optuna.samplers import TPESampler  

from vtbench.models.SimpleCNN import Simple2DCNN
from vtbench.models.DeepCNN import Deep2DCNN
from vtbench.CNN_train import create_dataloaders, train_model, evaluate_model

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

excel_file_path = 'bayesian_optim_optuna_results.xlsx'

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

    model_class = Simple2DCNN if model_name == 'Simple2DCNN' else Deep2DCNN
    model = model_class(3, nb_classes).to(device)

    if apply_smote_flag:
        desired_samples_per_class = {  
            1: 200,
            2: 200,
            3: 250,
            4: 300
        }
        x_train_resampled, y_train_resampled = apply_smote(x_train, y_train, desired_samples_per_class)
        X_train, y_train, X_test, y_test = to_torch_tensors(x_train_resampled, y_train_resampled, x_test, y_test)
    else:
        X_train, y_train, X_test, y_test = to_torch_tensors(x_train, y_train, x_test, y_test)

    print(f"SMOTE applied: {apply_smote_flag} | Shape of X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Shape of X_test: {X_test.shape}, y_test: {y_test.shape}")

    dataloaders = create_dataloaders(X_train, y_train, X_test, y_test)

    train_loader, val_loader, test_loader = dataloaders[combo_key]

    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

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

    metrics = evaluate_model(model, test_loader)

    results = {
        'SMOTE': apply_smote_flag,
        'Model': model_name,
        'Combination': combo_key,
        'Best Validation Loss': best_val_loss,
        'Best Validation Accuracy': best_val_accuracy,
        'Test Loss': metrics['test_loss'],
        'Test Accuracy': metrics['test_accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1 Score': metrics['f1_score'],
        'AUC': metrics['auc'],
        'Balanced Accuracy': metrics['balanced_accuracy'],
        'Specificity': metrics['avg_specificity'],
        'Learning Rate': lr,
        'Weight Decay': weight_decay,
        'Patience': patience,
        'Batch Size': batch_size,
        'Optimizer': optimizer_type,
        'Dropout Rate': dropout_rate,
        'Kernel Size': kernel_size
    }

    write_results_to_excel(excel_file_path, results, f"Bayesian Optimization")

    return metrics['balanced_accuracy']

sampler = TPESampler()  
study = optuna.create_study(direction='maximize', sampler=sampler)


study.optimize(objective, n_trials=100)  

print("Best Hyperparameters: ", study.best_params)
