import argparse
import torch
from torch import nn
from num.models.LSTM import LSTMModel
from num.LSTM_train import create_dataloader, stratified_split, train_model, validate_model, test_model
from num.data_utils import read_ucr, normalize_data, to_torch_tensors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    X_train, y_train = read_ucr(args.train_file)
    X_test, y_test = read_ucr(args.test_file)

    
    X_train, X_test = normalize_data(X_train, X_test)
    X_train, y_train, X_test, y_test = to_torch_tensors(X_train, y_train, X_test, y_test)

    
    X_train, X_val, y_train, y_val = stratified_split(X_test, y_test)

    
    train_loader = create_dataloader(X_train, y_train)
    val_loader = create_dataloader(X_val, y_val)
    test_loader = create_dataloader(X_test, y_test)

    
    model = LSTMModel(input_size=1, hidden_size=100, num_layers=3).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
   

    
    epochs =200
    for epoch in range(epochs):
       
        print(f"Epoch {epoch + 1}/{epochs}")

        
        train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)

    
        
    print("\nTesting the model on the test dataset:")
    test_model(model, test_loader, device)


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LSTM on binary time series data.")
    parser.add_argument('--train_file', required=True, help="Path to the training dataset")
    parser.add_argument('--test_file', required=True, help="Path to the testing dataset")
    args = parser.parse_args()

    main(args)
