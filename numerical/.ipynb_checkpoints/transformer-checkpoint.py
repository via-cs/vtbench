import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix

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

train_file = '../data/ECG5000/ECG5000_TRAIN.ts'
test_file = '../data/ECG5000/ECG5000_TEST.ts'

# Load dataset
x_train, y_train = read_ucr(train_file)
x_test, y_test = read_ucr(test_file)

# Normalize labels
unique_labels = np.unique(y_train)
label_map = {label: idx for idx, label in enumerate(unique_labels)}
y_train = np.array([label_map[label] for label in y_train])
y_test = np.array([label_map[label] for label in y_test])

nb_classes = len(unique_labels)

smote = SMOTE(k_neighbors=1)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

X_train = torch.tensor(x_train_resampled, dtype=torch.float32).unsqueeze(1)  
X_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train_resampled, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=128, nhead=4, num_encoder_layers=3, dim_feedforward=256, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.d_model = d_model
        self.input_dim = input_dim
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = src.permute(2, 0, 1)  
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=0)  
        output = self.fc(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Initialize model
input_dim = X_train.shape[-1]
num_classes = nb_classes
model = TransformerModel(input_dim=input_dim, num_classes=num_classes).to(device)

# Define optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
def train_transformer(model, train_loader, criterion, optimizer, num_epochs, test_loader):
    model.train()
    for epoch in range(num_epochs):
        total_train_loss = 0
        correct = 0
        total = 0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = total_train_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        validation_accuracy, balanced_accuracy = evaluate_transformer(model, test_loader)

        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {validation_accuracy:.2f}%, Balanced Accuracy: {balanced_accuracy:.2f}%')

# Evaluation on test data
def evaluate_transformer(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # Raw logits
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = correct / total * 100

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Calculate recall and specificity for each class
    recall_per_class = []
    specificity_per_class = []

    for i in range(len(conf_matrix)):
        # Recall (Sensitivity) for class i: TP / (TP + FN)
        recall_i = conf_matrix[i, i] / conf_matrix[i, :].sum() if conf_matrix[i, :].sum() != 0 else 0
        recall_per_class.append(recall_i)
        
        # Specificity for class i: TN / (TN + FP)
        tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
        fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
        specificity_i = tn / (tn + fp) if (tn + fp) != 0 else 0
        specificity_per_class.append(specificity_i)
    
    # Calculate average recall and specificity
    avg_recall = sum(recall_per_class) / len(recall_per_class)
    avg_specificity = sum(specificity_per_class) / len(specificity_per_class)
    
    # Calculate balanced accuracy
    balanced_acc = (avg_recall + avg_specificity) / 2 * 100

    return accuracy, balanced_acc
# Train and evaluate the model
num_epochs = 100
train_transformer(model, train_loader, criterion, optimizer, num_epochs, test_loader)

# Final evaluation on the test set
test_accuracy, test_balanced_accuracy = evaluate_transformer(model, test_loader)
print(f"Final Test Accuracy: {test_accuracy:.2f}%")
print(f"Final Test Balanced Accuracy: {test_balanced_accuracy:.2f}%")
