import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from num.data_utils import read_ucr, read_ecg5000, normalize_data, to_torch_tensors

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_len, d_model=128, nhead=8, num_layers=4, num_classes=2):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, input_len)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.input_projection.bias.data.zero_()
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()

    def forward(self, src):
        # src shape: [batch_size, seq_len]
        src = src.unsqueeze(-1)  # [batch_size, seq_len, 1]
        src = self.input_projection(src) * np.sqrt(self.d_model)  # [batch_size, seq_len, d_model]
        src = src.transpose(0, 1)  # [seq_len, batch_size, d_model]
        src = self.pos_encoder(src)
        
        output = self.transformer_encoder(src)  # [seq_len, batch_size, d_model]
        
        # Global average pooling
        output = output.mean(dim=0)  # [batch_size, d_model]
        
        return self.classifier(output)

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Standard training without data augmentation"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    
    return total_loss / len(train_loader)

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Get probabilities for AUC calculation
            probs = F.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            total_loss += loss.item()
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    # AUC score (using probabilities of positive class)
    auc = roc_auc_score(all_labels, all_probs[:, 1])
    
    # Confusion matrix for specificity
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    avg_loss = total_loss / len(loader)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'specificity': specificity,
        'loss': avg_loss
    }

def main(args):
    # Fixed random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    dataset_folder = os.path.join("data", args.dataset)
    train_file = os.path.join(dataset_folder, f"{args.dataset}_TRAIN.ts")
    test_file = os.path.join(dataset_folder, f"{args.dataset}_TEST.ts")

    if args.dataset.lower() == "ecg5000":
        x_train, y_train = read_ecg5000(train_file)
        x_test, y_test = read_ecg5000(test_file)
    else:
        x_train, y_train, _ = read_ucr(train_file)
        x_test, y_test, _ = read_ucr(test_file)

    # Normalize data
    x_train, x_test = normalize_data(x_train, x_test)

    # Create validation set
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, stratify=y_train, random_state=1337
    )

    # Convert to tensors
    X_train, y_train, X_val, y_val = to_torch_tensors(x_train, y_train, x_val, y_val)
    X_test, y_test = torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)

    # Calculate class weights for balanced training
    from collections import Counter
    class_counts = Counter(y_train.numpy())
    total_samples = sum(class_counts.values())
    class_weights = torch.FloatTensor([
        total_samples / (len(class_counts) * class_counts[i]) 
        for i in sorted(class_counts.keys())
    ])
    
    print(f"Class distribution: {class_counts}")
    print(f"Class weights: {class_weights}")
    
    # Data loaders with balanced sampling
    from torch.utils.data import WeightedRandomSampler
    sample_weights = [class_weights[y] for y in y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, sampler=sampler)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    num_classes = len(set(y_train.numpy()))
    print(f"Number of classes: {num_classes}")
    print(f"Input length: {X_train.shape[1]}")
    
    # Create model
    model = TimeSeriesTransformer(
        input_len=X_train.shape[1],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        num_classes=num_classes
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Optimizer with different learning rates for different parts
    base_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'classifier' in name:
            classifier_params.append(param)
        else:
            base_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': args.lr},
        {'params': classifier_params, 'lr': args.lr * 2}  # Higher LR for classifier
    ], weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[args.lr, args.lr * 2],
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    print("\nStarting training...")
    print("=" * 80)
    
    # best_val_f1 = 0.0
    # patience = 
    # patience_counter = 0
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        test_metrics = evaluate(model, test_loader, device)
        
        # Step scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            # OneCycleLR steps every batch, not every epoch
            pass
        else:
            scheduler.step()
        
        # # Early stopping
        # if val_metrics['f1'] > best_val_f1:
        #     best_val_f1 = val_metrics['f1']
        #     patience_counter = 0
        #     torch.save(model.state_dict(), f'best_model_{args.dataset}.pth')
        # else:
        #     patience_counter += 1
        
        # Print progress
        if epoch % 10 == 0 or epoch < 20:
            current_lrs = [group['lr'] for group in optimizer.param_groups]
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f}")
            print(f"         Val - Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f} | "
                  f"AUC: {val_metrics['auc']:.4f} | Spec: {val_metrics['specificity']:.4f}")
            print(f"        Test - Acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1']:.4f} | "
                  f"AUC: {test_metrics['auc']:.4f} | Spec: {test_metrics['specificity']:.4f}")
            print(f"         LR: {current_lrs[0]:.2e}")
            print("-" * 80)
        
        # if patience_counter >= patience:
        #     print(f"\nEarly stopping at epoch {epoch+1}")
        #     break
    
    # Final evaluation
    print("\n" + "=" * 80)
    print("Final evaluation with best model:")
    model.load_state_dict(torch.load(f'best_model_{args.dataset}.pth', weights_only=True))
    
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)
    
    print(f"\nFINAL RESULTS:")
    print("=" * 50)
    print("VALIDATION METRICS:")
    print(f"  Accuracy:    {val_metrics['accuracy']:.4f}")
    print(f"  Precision:   {val_metrics['precision']:.4f}")
    print(f"  Recall:      {val_metrics['recall']:.4f}")
    print(f"  F1-Score:    {val_metrics['f1']:.4f}")
    print(f"  AUC:         {val_metrics['auc']:.4f}")
    print(f"  Specificity: {val_metrics['specificity']:.4f}")
    print()
    print("TEST METRICS:")
    print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"  Precision:   {test_metrics['precision']:.4f}")
    print(f"  Recall:      {test_metrics['recall']:.4f}")
    print(f"  F1-Score:    {test_metrics['f1']:.4f}")
    print(f"  AUC:         {test_metrics['auc']:.4f}")
    print(f"  Specificity: {test_metrics['specificity']:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    args = parser.parse_args()
    main(args)
    