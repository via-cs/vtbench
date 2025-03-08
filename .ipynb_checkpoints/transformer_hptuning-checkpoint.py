# import argparse
# import torch
# import optuna
# import json
# from num.transformer_train import prepare_dataloaders, create_train_val_split, train_model, evaluate_model
# from num.models.transformer import TransformerClassifier
# from num.data_utils import read_ucr, read_ecg5000, normalize_data, to_torch_tensors
# from torch.utils.data import DataLoader, TensorDataset

# def setup_model(input_dim, num_classes, device, learning_rate, d_model, nhead, num_layers, dim_feedforward, dropout):
#     model = TransformerClassifier(
#         input_dim=input_dim,
#         num_classes=num_classes,
#         d_model=d_model,
#         nhead=nhead,
#         num_layers=num_layers,
#         dim_feedforward=dim_feedforward,
#         dropout=dropout
#     ).to(device)
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
#     return model, criterion, optimizer

# def save_best_hyperparameters(dataset_name, best_trial):
#     """Save the best hyperparameters for each dataset."""
#     best_params = {
#         "dataset": dataset_name,
#         "value": best_trial.value,
#         "params": best_trial.params
#     }
#     try:
#         with open("best_hyperparameters.json", "r") as f:
#             best_results = json.load(f)
#     except (FileNotFoundError, json.JSONDecodeError):
#         best_results = {}

#     best_results[dataset_name] = best_params
#     with open("best_hyperparameters.json", "w") as f:
#         json.dump(best_results, f, indent=4)
#     print(f"Saved best hyperparameters for {dataset_name}")

# def save_best_transformer_config(dataset_name, best_trial):
#     """Save the best transformer architecture for each dataset."""
#     config_str = f"""import torch
# import torch.nn as nn

# class TransformerClassifier(nn.Module):
#     def __init__(self, input_dim, num_classes):
#         super(TransformerClassifier, self).__init__()
#         self.embedding = nn.Linear(input_dim, {best_trial.params['d_model']})
#         self.transformer = nn.Transformer(
#             d_model={best_trial.params['d_model']},
#             nhead={best_trial.params['nhead']},
#             num_encoder_layers={best_trial.params['num_layers']},
#             num_decoder_layers=0,
#             dim_feedforward={best_trial.params['dim_feedforward']},
#             dropout={best_trial.params['dropout']},
#             batch_first=True
#         )
#         self.fc = nn.Linear({best_trial.params['d_model']}, num_classes)

#     def forward(self, x):
#         if x.dim() == 2:
#             x = x.unsqueeze(-1)
#         batch_size, seq_len, _ = x.size()
#         positional_encoding = self.get_positional_encoding(seq_len, self.embedding.out_features, x.device)
#         x = self.embedding(x) + positional_encoding[:, :seq_len, :]
#         x = self.transformer.encoder(x)
#         return self.fc(x[:, -1, :])

#     @staticmethod
#     def get_positional_encoding(seq_len, d_model, device):
#         positions = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * (-torch.log(torch.tensor(10000.0)) / d_model))
#         pe = torch.zeros(seq_len, d_model, device=device)
#         pe[:, 0::2] = torch.sin(positions * div_term)
#         pe[:, 1::2] = torch.cos(positions * div_term)
#         return pe.unsqueeze(0)
# """
#     with open(f"transformer_{dataset_name.lower()}.py", "w") as f:
#         f.write(config_str)
#     print(f"Saved best Transformer architecture for {dataset_name}")

# def objective(trial, x_train, y_train, x_test, y_test):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     input_dim = x_train.shape[-1]
#     num_classes = len(torch.unique(y_train))

#     learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
#     batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
#     num_epochs = trial.suggest_int("num_epochs", 50, 200)
#     d_model = trial.suggest_categorical("d_model", [64, 128, 256, 512])
#     nhead = trial.suggest_categorical("nhead", [2, 4, 8, 16])
#     num_layers = trial.suggest_int("num_layers", 2, 6)
#     dim_feedforward = trial.suggest_categorical("dim_feedforward", [512, 1024, 2048])
#     dropout = trial.suggest_float("dropout", 0.1, 0.5)

   
#     x_train, x_val, y_train, y_val = create_train_val_split(x_train, y_train, val_size=0.2)

#     model, criterion, optimizer = setup_model(
#         input_dim, num_classes, device, learning_rate, d_model, nhead, num_layers, dim_feedforward, dropout
#     )

#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)
    
#     train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

#     _, _, val_acc = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs)

#     val_acc = float(val_acc)  
#     print(f"Validation Accuracy for trial: {val_acc:.4f}")

#     return val_acc

# def main(train_file, test_file):
#     dataset_name = train_file.split("/")[-2]  
#     if dataset_name.lower() == "ecg5000":
#         x_train, y_train = read_ecg5000(train_file)
#         x_test, y_test = read_ecg5000(test_file)
#     else:
#         x_train, y_train = read_ucr(train_file)
#         x_test, y_test = read_ucr(test_file)
#     x_train, x_test = normalize_data(x_train, x_test)
#     X_train, y_train, X_test, y_test = to_torch_tensors(x_train, y_train, x_test, y_test)

#     study = optuna.create_study(direction="maximize")
#     study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=100)

#     print("Best trial:")
#     trial = study.best_trial
#     print(f"Validation Accuracy: {trial.value}")
#     print("Params:")
#     for key, value in trial.params.items():
#         print(f"    {key}: {value}")

#     save_best_hyperparameters(dataset_name, trial)
#     save_best_transformer_config(dataset_name, trial)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Hyperparameter tuning for Transformer model on UCR datasets")
#     parser.add_argument("--train_file", type=str, required=True, help="Path to the training .ts file")
#     parser.add_argument("--test_file", type=str, required=True, help="Path to the testing .ts file")
#     args = parser.parse_args()

#     main(args.train_file, args.test_file)

import argparse
import torch
import optuna
import json
import pickle
from num.transformer_train import prepare_dataloaders, create_train_val_split, train_model, evaluate_model
from num.models.transformer import TransformerClassifier
from num.data_utils import read_ucr, read_ecg5000, normalize_data, to_torch_tensors
from torch.utils.data import DataLoader, TensorDataset

def setup_model(input_dim, num_classes, device, learning_rate, d_model, nhead, num_layers, dim_feedforward, dropout):
    model = TransformerClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    return model, criterion, optimizer

def save_best_hyperparameters(dataset_name, best_trial):
    """Save the best hyperparameters for each dataset."""
    best_params = {
        "dataset": dataset_name,
        "value": best_trial.value,
        "params": best_trial.params
    }
    try:
        with open("best_hyperparameters.json", "r") as f:
            best_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        best_results = {}

    best_results[dataset_name] = best_params
    with open("best_hyperparameters.json", "w") as f:
        json.dump(best_results, f, indent=4)
    print(f"Saved best hyperparameters for {dataset_name}")

def save_best_transformer_config(dataset_name, best_trial):
    """Save the best transformer architecture for each dataset."""
    config_str = f"""import torch
import torch.nn as nn

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, {best_trial.params['d_model']})
        self.transformer = nn.Transformer(
            d_model={best_trial.params['d_model']},
            nhead={best_trial.params['nhead']},
            num_encoder_layers={best_trial.params['num_layers']},
            num_decoder_layers=0,
            dim_feedforward={best_trial.params['dim_feedforward']},
            dropout={best_trial.params['dropout']},
            batch_first=True
        )
        self.fc = nn.Linear({best_trial.params['d_model']}, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        batch_size, seq_len, _ = x.size()
        positional_encoding = self.get_positional_encoding(seq_len, self.embedding.out_features, x.device)
        x = self.embedding(x) + positional_encoding[:, :seq_len, :]
        x = self.transformer.encoder(x)
        return self.fc(x[:, -1, :])

    @staticmethod
    def get_positional_encoding(seq_len, d_model, device):
        positions = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        return pe.unsqueeze(0)
"""
    with open(f"transformer_{dataset_name.lower()}.py", "w") as f:
        f.write(config_str)
    print(f"Saved best Transformer architecture for {dataset_name}")

def objective(trial, x_train, y_train, x_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if x_train.dim() == 2:
        x_train = x_train.unsqueeze(-1)  # Convert (batch, seq_len) -> (batch, seq_len, 1)
    if x_test.dim() == 2:
        x_test = x_test.unsqueeze(-1)

    input_dim = x_train.shape[-1]
    num_classes = torch.unique(y_train, return_counts=False).numel()

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    num_epochs = trial.suggest_int("num_epochs", 50, 200)
    d_model = trial.suggest_categorical("d_model", [64, 128, 256, 512])
    nhead = trial.suggest_categorical("nhead", [2, 4, 8, 16])
    num_layers = trial.suggest_int("num_layers", 2, 6)
    dim_feedforward = trial.suggest_categorical("dim_feedforward", [512, 1024, 2048])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    x_train, x_val, y_train, y_val = create_train_val_split(x_train, y_train, val_size=0.2)

    model, criterion, optimizer = setup_model(
        input_dim, num_classes, device, learning_rate, d_model, nhead, num_layers, dim_feedforward, dropout
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)
    
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

    _, _, val_acc = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs)

    return float(val_acc)

def main(train_file, test_file):
    dataset_name = train_file.split("/")[-2]
    
    if dataset_name.lower() == "ecg5000":
        x_train, y_train = read_ecg5000(train_file)
        x_test, y_test = read_ecg5000(test_file)
    else:
        x_train, y_train = read_ucr(train_file)
        x_test, y_test = read_ucr(test_file)

    x_train, x_test = normalize_data(x_train, x_test)
    X_train, y_train, X_test, y_test = to_torch_tensors(x_train, y_train, x_test, y_test)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=100)

    trial = study.best_trial
    print(f"Best Validation Accuracy: {trial.value}")
    print("Best Parameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    save_best_hyperparameters(dataset_name, trial)
    save_best_transformer_config(dataset_name, trial)

    best_params = trial.params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, criterion, optimizer = setup_model(
        X_train.shape[-1], torch.unique(y_train, return_counts=False).numel(), device,
        best_params["learning_rate"], best_params["d_model"], best_params["nhead"],
        best_params["num_layers"], best_params["dim_feedforward"], best_params["dropout"]
    )

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=best_params["batch_size"], shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=best_params["batch_size"], shuffle=False)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)

    train_model(model, train_loader, None, criterion, optimizer, scheduler, device, best_params["num_epochs"])

    test_acc = evaluate_model(model, test_loader, device)
    print(f"Final Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for Transformer model on UCR datasets")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training .ts file")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the testing .ts file")
    args = parser.parse_args()

    main(args.train_file, args.test_file)
