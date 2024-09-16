import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F 
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
X_train = torch.tensor(x_train_resampled, dtype=torch.float32).unsqueeze(1)  # Add channel dimension before sequence length
X_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train_resampled, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create Data Loaders
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define a Residual Block with TCN layers
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation  # Causal padding to maintain input-output size
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Slightly increased dropout for regularization
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        residual = self.residual_conv(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        # Slice x to match the residual size in case padding creates mismatches
        if x.size(2) != residual.size(2):
            x = x[:, :, :residual.size(2)]
        return self.relu(x + residual)  # Skip connection


# Define the complete TCN model
class TCN(nn.Module):
    def __init__(self, input_size, num_classes, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, dilation=dilation_size))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
    
    def forward(self, x):
        x = self.network(x)
        x = torch.mean(x, dim=-1)  # Global average pooling
        x = self.fc(x)
        return F.log_softmax(x, dim=1)  # Apply log_softmax for NLLLoss

input_size = 1  # Input size is 1 since we now have a single channel
num_classes = len(np.unique(y_train_resampled))
num_channels = [64, 128, 256]  # Adjust based on experimentation
model = TCN(input_size=input_size, num_classes=num_classes, num_channels=num_channels).to(device)

# Define optimizer and loss function
criterion = nn.NLLLoss()  # NLL Loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)


train_losses = []
validation_accuracies = []
test_accuracies = []
balanced_accuracies = []

# Training the model
def train_tcn(model, train_loader, criterion, optimizer, num_epochs, test_loader):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Calculate and store metrics
        train_loss = total_train_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        validation_accuracy, balanced_accuracy = evaluate_tcn(model, test_loader)
        
        train_losses.append(train_loss)
        validation_accuracies.append(validation_accuracy)
        balanced_accuracies.append(balanced_accuracy)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {validation_accuracy:.2f}%, Test Balanced Accuracy: {balanced_accuracy:.4f}')

# Evaluation on test data
def evaluate_tcn(model, data_loader):
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

num_epochs = 500
# Train the model
train_tcn(model, train_loader, criterion, optimizer, num_epochs, test_loader)

# Final evaluation on the test set
test_accuracy, test_balanced_accuracy = evaluate_tcn(model, test_loader)
print(f"Final Test Accuracy: {test_accuracy:.2f}%")
print(f"Final Test Balanced Accuracy: {test_balanced_accuracy:.4f}")
