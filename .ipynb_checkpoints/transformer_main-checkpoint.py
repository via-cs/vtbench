import argparse
import json
import torch
from num.transformer_train import prepare_dataloaders, train_model, evaluate_model
from num.data_utils import read_ucr, read_ecg5000, normalize_data, to_torch_tensors
from torch.utils.data import DataLoader, TensorDataset
from num.models.transformer import TransformerClassifier

def load_best_hyperparameters(dataset_name):
    with open("best_hyperparameters.json", "r") as f:
        best_results = json.load(f)
    return best_results.get(dataset_name, None)

def setup_model(input_dim, num_classes, device, best_params):
    model = TransformerClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        d_model=best_params['d_model'],
        nhead=best_params['nhead'],
        num_layers=best_params['num_layers'],
        dim_feedforward=best_params['dim_feedforward'],
        dropout=best_params['dropout']
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= 0.004, weight_decay=1e-4)
    return model, criterion, optimizer

def main(train_file, test_file):
    dataset_name = train_file.split("/")[-2]  
    best_params = load_best_hyperparameters(dataset_name)
    if best_params is None:
        raise ValueError(f"No saved hyperparameters found for {dataset_name}. Run tuning first.")

    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    if dataset_name.lower() == "ecg5000":
        x_train, y_train = read_ecg5000(train_file)
        x_test, y_test = read_ecg5000(test_file)
    else:
        x_train, y_train = read_ucr(train_file)
        x_test, y_test = read_ucr(test_file)
    x_train, x_test = normalize_data(x_train, x_test)
    X_train, y_train, X_test, y_test = to_torch_tensors(x_train, y_train, x_test, y_test)

    input_dim = 1
    num_classes = len(torch.unique(y_train))
    model, criterion, optimizer = setup_model(input_dim, num_classes, device, best_params['params'])
    
    batch_size = best_params['params']['batch_size']
    num_epochs = 500 #best_params['params']['num_epochs']   
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
    train_model(model, train_loader, None, criterion, optimizer, None, device, num_epochs)
    
    metrics = evaluate_model(model, test_loader, device, return_metrics=True)
    print("Final Test Evaluation:")
    print(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate Transformer model on UCR datasets")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training .ts file")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the testing .ts file")
    args = parser.parse_args()
    main(args.train_file, args.test_file)
