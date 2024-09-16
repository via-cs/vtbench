# Import necessary libraries
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from vtbench.data_utils import read_ucr, normalize_data, apply_smote, to_torch_tensors
import optuna
import os

from vtbench.models.SimpleCNN import Simple2DCNN
from vtbench.models.DeepCNN import Deep2DCNN

from vtbench.CNN_train import create_dataloaders, train_model, evaluate_model, plot_class_distribution

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# File paths
train_file = 'data/ECG5000/ECG5000_TRAIN.ts'
test_file = 'data/ECG5000/ECG5000_TEST.ts'

# Load datasets (Declare as global)
global x_train, y_train, x_test, y_test
x_train, y_train = read_ucr(train_file)
x_test, y_test = read_ucr(test_file)

print(f"Initial x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"Initial x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

# Normalize labels to be within range [0, num_classes-1]
unique_labels = np.unique(y_train)
label_map = {label: idx for idx, label in enumerate(unique_labels)}
y_train = np.array([label_map[label] for label in y_train])
y_test = np.array([label_map[label] for label in y_test])

nb_classes = len(unique_labels)

# Normalize features
x_train, x_test = normalize_data(x_train, x_test)

print(f"After normalization x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")


if x_train is None or y_train is None or x_test is None or y_test is None:
    raise ValueError("Training or testing data is not properly initialized. Please check your data loading.")


excel_file_path = 'hyperparameter_results.xlsx'


def write_results_to_excel(file_path, results, version):
    # Load the workbook or create a new one if it doesn't exist
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        df = pd.DataFrame()

    # Convert results to a DataFrame
    results_df = pd.DataFrame([results])

    # Add the version column
    results_df['Version'] = version

    # Concatenate the new results with the existing DataFrame
    df = pd.concat([df, results_df], ignore_index=True)

    # Save the DataFrame back to the Excel file
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
        df.to_excel(writer, index=False)



def objective(trial):

    global x_train, y_train, x_test, y_test

    
    x_train, y_train = read_ucr(train_file)
    x_test, y_test = read_ucr(test_file)

    # Normalize labels to be within range [0, num_classes-1]
    unique_labels = np.unique(y_train)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y_train = np.array([label_map[label] for label in y_train])
    y_test = np.array([label_map[label] for label in y_test])

    nb_classes = len(unique_labels)

    # Normalize features
    x_train, x_test = normalize_data(x_train, x_test)

    
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 3e-4, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-3, 3e-3, log=True)
    model_name = trial.suggest_categorical('model', ['Simple2DCNN', 'Deep2DCNN'])
    patience = trial.suggest_int('patience', 6, 10)
    apply_smote_flag = trial.suggest_categorical('apply_smote', [True, False])
    combo_key = trial.suggest_categorical('combo_key', list(create_dataloaders(x_train, y_train, x_test, y_test).keys()))
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    optimizer_type = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
    dropout_rate = trial.suggest_float('dropout_rate', 0.4, 0.5)
    kernel_size = trial.suggest_categorical('kernel_size', [2, 7])

    print(f"Running combination: Model={model_name}, Apply SMOTE={apply_smote_flag}, Combination={combo_key}, Batch Size={batch_size}, Optimizer={optimizer_type}, Dropout Rate={dropout_rate}, Kernel Size={kernel_size}, Learning Rate={lr}, Weight Decay={weight_decay}, Patience={patience}")

    # Select the model class based on the suggested model name
    model_class = Simple2DCNN if model_name == 'Simple2DCNN' else Deep2DCNN
    model = model_class(3, nb_classes).to(device)


    # Apply SMOTE if the flag is set
    if apply_smote_flag:
        desired_samples_per_class = {
            1: 200,
            2: 100,
            3: 50,
            4: 50
        }
        x_train_resampled, y_train_resampled = apply_smote(x_train, y_train, desired_samples_per_class)
        X_train, y_train, X_test, y_test = to_torch_tensors(x_train_resampled, y_train_resampled, x_test, y_test)
    else:
        # Use original data without applying SMOTE
        X_train, y_train, X_test, y_test = to_torch_tensors(x_train, y_train, x_test, y_test)

    print(f"SMOTE applied: {apply_smote_flag} | Shape of X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Shape of X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Create dataloaders
    dataloaders = create_dataloaders(X_train, y_train, X_test, y_test)
    
    # Use the selected combo_key
    train_loader, val_loader, test_loader = dataloaders[combo_key]

    # Define optimizer and scheduler using suggested hyperparameters
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Initialize scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

    # Train the model with suggested hyperparameters
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

    # Evaluate the model
    metrics = evaluate_model(model, test_loader)
    
    # Collect results
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

    # Save the results to the Excel sheet for each trial
    write_results_to_excel(excel_file_path, results, f"Trial {trial.number + 1}")

    # Return the metrics to be optimized
    return metrics['balanced_accuracy']




# Run Optuna optimization
n_trials = 200
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=n_trials)

# Save the best hyperparameters
best_hyperparameters = study.best_params
print("Best Hyperparameters: ", best_hyperparameters)

