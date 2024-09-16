import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import balanced_accuracy_score


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read UCR Dataset
def read_ucr(filename):
    data = []
    labels = []
    
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            features = [float(f) for f in parts[:-1]]
            label = int(parts[-1].split(':')[-1])
            data.append(features)
            labels.append(label)
    
    print(f"Loaded {len(data)} samples from {filename}")
    return np.array(data), np.array(labels)

train_file = 'data/ECG5000/ECG5000_TRAIN.ts'
test_file = 'data/ECG5000/ECG5000_TEST.ts'

# Load dataset
x_train, y_train = read_ucr(train_file)
x_test, y_test = read_ucr(test_file)

# Normalize labels
unique_labels = np.unique(y_train)
label_map = {label: idx for idx, label in enumerate(unique_labels)}
y_train = np.array([label_map[label] for label in y_train])
y_test = np.array([label_map[label] for label in y_test])

nb_classes = len(unique_labels)

# Apply SMOTE to the training data
smote = SMOTE(k_neighbors=1)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

# Convert to PyTorch tensors
X_train = torch.tensor(x_train_resampled, dtype=torch.float32).unsqueeze(-1)
X_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train_resampled, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create Data Loaders
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(BiLSTMModel, self).__init__()
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                            batch_first=True, bidirectional=True, dropout=dropout)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size * 2, output_size)  # * 2 for bidirectionality
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Take the output of the last time step
        lstm_out = lstm_out[:, -1, :]  # Get the last output from the sequence
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Fully connected layer
        output = self.fc(lstm_out)
        
        return output


# Hyperparameters
input_size = 1  # ECG time series input
hidden_size = 128  # Number of LSTM units
output_size = nb_classes  # Number of classes (5 for ECG5000)
num_layers = 2  # Number of LSTM layers
dropout = 0.5  # Dropout rate

# Initialize model, loss, optimizer
model = BiLSTMModel(input_size, hidden_size, output_size, num_layers, dropout).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Tracking metrics
train_losses = []
validation_accuracies = []
test_accuracies = []
balanced_accuracies = []

# Training the model
def train_lstm(model, train_loader, criterion, optimizer, num_epochs, test_loader):
    model.train()
    for epoch in range(num_epochs):
        total_train_loss = 0
        correct = 0
        total = 0
        
        # Training loop
        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_train_loss += loss.item()
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Calculate and store metrics
        train_loss = total_train_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        validation_accuracy, balanced_accuracy = evaluate_lstm(model, test_loader)
        
        train_losses.append(train_loss)
        validation_accuracies.append(validation_accuracy)
        balanced_accuracies.append(balanced_accuracy)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {validation_accuracy:.2f}%, Test balanced Accuracy: {balanced_accuracy:.4f}')
        
# Evaluation on test data
def evaluate_lstm(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = correct / total * 100
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    return accuracy, balanced_acc

num_epochs = 200
# Train the model
train_lstm(model, train_loader, criterion, optimizer, num_epochs, test_loader)

# Final evaluation on the test set
test_accuracy, test_balanced_accuracy = evaluate_lstm(model, test_loader)
print(f"Final Test Accuracy: {test_accuracy:.2f}%")
print(f"Final Test Balanced Accuracy: {test_balanced_accuracy:.4f}")