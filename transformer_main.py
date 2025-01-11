import argparse
import torch
import torch.nn as nn
from num.transformer_train import prepare_dataloaders, train_model, evaluate_model
from num.models.Transformer import Transformer
from num.data_utils import read_ucr, normalize_data, to_torch_tensors  

def run_training(x_train, y_train, x_test, y_test, num_epochs, batch_size, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = 1  
    num_classes = len(set(y_train.numpy()))

    model = TransformerClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    train_loader, test_loader = prepare_dataloaders(x_train, y_train, x_test, y_test, batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    train_model(model, train_loader, criterion, optimizer, device, num_epochs)
    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a Transformer model on UCR datasets")
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training .ts file")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the testing .ts file")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    x_train, y_train = read_ucr(args.train_path)
    x_test, y_test = read_ucr(args.test_path)
    x_train, x_test = normalize_data(x_train, x_test)
    X_train, y_train, X_test, y_test = to_torch_tensors(x_train, y_train, x_test, y_test)

    run_training(X_train, y_train, X_test, y_test, num_epochs=args.num_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)
