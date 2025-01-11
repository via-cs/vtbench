import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.3):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=0,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        if x.dim() == 2:  
            x = x.unsqueeze(-1)  
    
        batch_size, seq_len, _ = x.size()
        d_model = self.embedding.out_features
    
        positional_encoding = self.get_positional_encoding(seq_len, d_model, x.device)
    
        x = self.embedding(x) + positional_encoding[:, :seq_len, :]
        
        x = self.transformer.encoder(x) 
        x = x[:, -1, :]  
        return self.fc(x)


    @staticmethod
    def get_positional_encoding(seq_len, d_model, device):
        """
        Generate positional encoding dynamically based on sequence length and model dimension.
        """
        positions = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        return pe.unsqueeze(0)  


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=20):
    model.train()
    epoch_losses = []  
    epoch_accuracies = []  

    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        avg_loss = epoch_loss / len(train_loader)
        accuracy = correct / total

        epoch_losses.append(avg_loss)
        epoch_accuracies.append(accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")

    return epoch_losses, epoch_accuracies




def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy


def main(x_train, y_train, x_test, y_test, num_epochs=50, batch_size=16, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    input_dim = 1  
    num_classes = len(set(y_train.numpy()))
    model = TransformerClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = TransformerClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    train_model(model, train_loader, criterion, optimizer, device, num_epochs=num_epochs)
    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    import argparse
    from num.data_utils import read_ucr, normalize_data, to_torch_tensors, apply_smote  

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

    main(X_train, y_train, X_test, y_test, num_epochs=args.num_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)
