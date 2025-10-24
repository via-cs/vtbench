#!/usr/bin/env python

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import time
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import math

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

warnings.filterwarnings('ignore')

# =============================================================================
# Chart Generation
# =============================================================================

class PixelPerDataAnalyzer:
    """Analyzes optimal pixel density for chart generation"""
    def __init__(self, target_ppd=0.7, content_ratio=0.95):
        self.target_ppd = target_ppd
        self.content_ratio = content_ratio
    
    def get_optimal_resolution(self, data_length):
        ideal_width = math.ceil(data_length * self.target_ppd / self.content_ratio)
        standard_widths = [64, 128, 256, 512]
        optimal_width = min([w for w in standard_widths if w >= ideal_width], default=512)
        
        if data_length <= 512:
            optimal_height = optimal_width
        else:
            aspect_ratio = min(4.0, max(1.0, data_length / 512))
            optimal_height = max(256, int(optimal_width / aspect_ratio))
        
        return optimal_height, optimal_width

class EnhancedImageGenerator:
    """Generates chart images from time series data"""
    def __init__(self, height, width, color_mode='color', label_mode='with_label'):
        self.height = height
        self.width = width
        self.color_mode = color_mode
        self.label_mode = label_mode
    
    def generate_image(self, time_series, chart_type='line'):
        try:
            ts_clean = np.nan_to_num(time_series.astype(np.float32), nan=0.0)
            return self._create_chart(ts_clean, chart_type)
        except Exception:
            return np.ones((3, self.height, self.width), dtype=np.float32)
    
    def _create_chart(self, ts, chart_type):
        fig_width = self.width / 100
        fig_height = self.height / 100
        plt.figure(figsize=(fig_width, fig_height), dpi=100, facecolor='white')
        
        x_data = np.arange(len(ts))
        color = 'blue' if self.color_mode == 'color' else 'black'
        
        if chart_type == 'area':
            plt.fill_between(x_data, ts, color=color)
        elif chart_type == 'line':
            plt.plot(ts, color=color)
        elif chart_type == 'scatter':
            plt.scatter(x_data, ts, color=color)
        elif chart_type == 'bar':
            plt.bar(x_data, ts, color='none', edgecolor=color, width=1.0)
        
        if self.label_mode == 'with_label':
            chart_titles = {'area': 'Area Chart', 'line': 'Line Chart', 
                          'scatter': 'Scatter Chart', 'bar': 'Bar Chart'}
            plt.title(chart_titles.get(chart_type, 'Chart'))
        else:
            plt.axis('off')
        
        image = self._convert_to_array()
        plt.close()
        return image
    
    def _convert_to_array(self):
        from io import BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                   pad_inches=0, facecolor='white')
        buf.seek(0)
        pil_img = Image.open(buf).convert('RGB').resize((self.width, self.height))
        image = np.array(pil_img).astype(np.float32) / 255.0
        buf.close()
        return image.transpose(2, 0, 1)

def load_ucr_data(file_path):
    """Load UCR dataset from TSV file"""
    time_series_list = []
    labels = []
    
    with open(file_path, 'r') as f:
        for line in f:
            values = line.strip().split('\t')
            if len(values) < 2:
                continue
            label = int(float(values[0]))
            ts_data = np.array([float(v) for v in values[1:] if v != 'NaN'])
            if len(ts_data) > 0:
                labels.append(label)
                time_series_list.append(ts_data)
    
    return time_series_list, labels

def normalize_labels(labels):
    """Normalize labels to start from 0"""
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    normalized_labels = np.array([label_mapping[label] for label in labels])
    return normalized_labels.tolist()

def generate_charts_for_dataset(dataset_name, ucr_data_dir, output_dir, chart_types):
    """Generate all chart types for a dataset"""
    print(f"Generating charts: {dataset_name}")
    
    dataset_path = Path(ucr_data_dir) / dataset_name
    train_file = dataset_path / f"{dataset_name}_TRAIN.tsv"
    test_file = dataset_path / f"{dataset_name}_TEST.tsv"
    
    train_data, train_labels = load_ucr_data(train_file)
    test_data, test_labels = load_ucr_data(test_file)
    
    train_labels = normalize_labels(train_labels)
    test_labels = normalize_labels(test_labels)
    
    print(f"  Train samples: {len(train_data)}, Test samples: {len(test_data)}")
    
    max_length = max(len(ts) for ts in train_data + test_data)
    analyzer = PixelPerDataAnalyzer()
    height, width = analyzer.get_optimal_resolution(max_length)
    print(f"  Image size: {height}×{width}")
    
    for chart_type in chart_types:
        print(f"  Generating {chart_type} charts...")
        
        chart_dir = Path(output_dir) / dataset_name / f"{chart_type}_charts_color_with_label"
        train_dir = chart_dir / "train"
        test_dir = chart_dir / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        generator = EnhancedImageGenerator(height, width, 'color', 'with_label')
        
        for i, ts in enumerate(train_data):
            img = generator.generate_image(ts, chart_type)
            save_path = train_dir / f"{chart_type}_{i}.png"
            img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(img_uint8.transpose(1, 2, 0)).save(save_path)
        
        for i, ts in enumerate(test_data):
            img = generator.generate_image(ts, chart_type)
            save_path = test_dir / f"{chart_type}_{i}.png"
            img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(img_uint8.transpose(1, 2, 0)).save(save_path)
    
    print("Chart generation complete")
    return train_labels, test_labels

# =============================================================================
# Model Definition
# =============================================================================

class ImprovedDeep2DCNN(nn.Module):
    """Improved 2D CNN model for chart classification"""
    def __init__(self, num_classes, input_size=256):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class SimpleDataset(Dataset):
    """Simple image dataset loader"""
    def __init__(self, image_dir, labels, transform=None):
        self.image_dir = Path(image_dir)
        self.image_files = sorted(list(self.image_dir.glob("*.png")))
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return min(len(self.image_files), len(self.labels))
    
    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

# =============================================================================
# Training
# =============================================================================

def train_single_model(train_loader, val_loader, num_classes, epochs, device, input_size=256):
    """Train a single model"""
    model = ImprovedDeep2DCNN(num_classes, input_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.003, epochs=epochs, steps_per_epoch=len(train_loader)
    )
    
    best_val_acc = 0
    patience_counter = 0
    patience = 10
    
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        if epoch % 5 == 0:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = 100 * val_correct / val_total
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
    
    return model

def train_all_models(dataset_name, chart_dir, train_labels, test_labels, 
                    chart_types, model_dir, epochs, device, input_size=256):
    """Train models for all chart types"""
    print(f"\nTraining models: {dataset_name}")
    
    num_classes = len(set(train_labels))
    print(f"  Number of classes: {num_classes}")
    
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    models = {}
    
    for chart_type in chart_types:
        print(f"  Training {chart_type} model...")
        
        chart_subdir = Path(chart_dir) / dataset_name / f"{chart_type}_charts_color_with_label"
        train_dir = chart_subdir / "train"
        test_dir = chart_subdir / "test"
        
        if not train_dir.exists() or not test_dir.exists():
            print(f"  Skipping {chart_type} (directory not found)")
            continue
        
        train_dataset = SimpleDataset(train_dir, train_labels, transform)
        test_dataset = SimpleDataset(test_dir, test_labels, transform)
        
        val_size = max(1, len(test_dataset) // 5)
        test_size = len(test_dataset) - val_size
        test_dataset, val_dataset = torch.utils.data.random_split(
            test_dataset, [test_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        model = train_single_model(train_loader, val_loader, num_classes, epochs, device, input_size)
        
        model_path = Path(model_dir) / dataset_name / f"{chart_type}_model.pth"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        
        models[chart_type] = (model, test_loader)
        print(f"  {chart_type} model training complete")
    
    return models

# =============================================================================
# Representation Analysis
# =============================================================================

class CKA:
    """Centered Kernel Alignment (CKA) similarity computation"""
    @staticmethod
    def centering(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones([n, n]) / n
        return np.dot(np.dot(H, K), H)
    
    @staticmethod
    def linear_CKA(X, Y):
        L_X = X @ X.T
        L_Y = Y @ Y.T
        hsic_xy = np.sum(CKA.centering(L_X) * CKA.centering(L_Y))
        hsic_xx = np.sum(CKA.centering(L_X) ** 2)
        hsic_yy = np.sum(CKA.centering(L_Y) ** 2)
        return hsic_xy / np.sqrt(hsic_xx * hsic_yy)

class FeatureExtractor:
    """Extract features from trained models"""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.features = None
        target_layer = list(self.model.fc_layers.children())[-2]
        target_layer.register_forward_hook(lambda m, i, o: setattr(self, 'features', o.detach()))
    
    def extract_features(self, dataloader):
        self.model.eval()
        all_features = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Extracting features", leave=False):
                inputs = inputs.to(self.device)
                _ = self.model(inputs)
                all_features.append(self.features.cpu().numpy())
                all_labels.append(labels.numpy())
        return np.vstack(all_features), np.concatenate(all_labels)

def compute_intrinsic_dimensionality(features_dict, chart_types, variance_threshold=0.90):
    """Compute intrinsic dimensionality using PCA"""
    results = {}
    
    for chart_type in chart_types:
        features, labels = features_dict[chart_type]
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        pca = PCA()
        pca.fit(features_scaled)
        
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        
        results[chart_type] = {
            'n_components_90': n_components,
            'variance_90': cumulative_variance[n_components-1],
            'total_components': len(cumulative_variance),
            'explained_variance_ratio': pca.explained_variance_ratio_[:20],
            'cumulative_variance': cumulative_variance[:20]
        }
    
    return results

def create_umap_visualizations(features_dict, chart_types, dataset_name, output_dir):
    """Create UMAP visualizations"""
    if not UMAP_AVAILABLE:
        print("  UMAP not available, skipping visualization")
        return
    
    all_features = []
    all_labels = []
    all_chart_types = []
    
    for chart_type in chart_types:
        features, labels = features_dict[chart_type]
        all_features.append(features)
        all_labels.append(labels)
        all_chart_types.extend([chart_type] * len(features))
    
    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)
    
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric='euclidean',
        random_state=42,
        n_jobs=1
    )
    
    embedding = reducer.fit_transform(all_features)
    
    # Visualization 1: Colored by true class
    fig, ax = plt.subplots(figsize=(12, 10))
    unique_labels = np.unique(all_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for idx, label in enumerate(unique_labels):
        mask = all_labels == label
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                  c=[colors[idx]], label=f'Class {label}',
                  alpha=0.6, s=20, edgecolors='none')
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title(f'UMAP: Colored by True Class - {dataset_name}', 
                fontweight='bold', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'umap_by_class_{dataset_name}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Visualization 2: Colored by chart type
    fig, ax = plt.subplots(figsize=(12, 10))
    chart_colors = {'line': '#1f77b4', 'area': '#ff7f0e', 
                   'scatter': '#2ca02c', 'bar': '#d62728'}
    
    for chart_type in chart_types:
        mask = np.array(all_chart_types) == chart_type
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                  c=chart_colors.get(chart_type, 'gray'),
                  label=f'{chart_type.capitalize()}',
                  alpha=0.6, s=20, edgecolors='none')
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title(f'UMAP: Colored by Chart Type - {dataset_name}', 
                fontweight='bold', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'umap_by_chart_{dataset_name}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def run_analysis(models, dataset_name, output_dir, chart_types, device):
    """Run comprehensive representation analysis"""
    print(f"\nRepresentation analysis: {dataset_name}")
    
    # Extract features
    features_with_labels = {}
    for chart_type, (model, test_loader) in models.items():
        extractor = FeatureExtractor(model, device)
        features, labels = extractor.extract_features(test_loader)
        features_with_labels[chart_type] = (features, labels)
    
    # Experiment 1A: CKA matrix
    print("\nExperiment 1A: CKA Similarity Analysis")
    n = len(chart_types)
    cka_matrix = np.zeros((n, n))
    for i, c1 in enumerate(chart_types):
        for j, c2 in enumerate(chart_types):
            if i <= j:
                feat1 = features_with_labels[c1][0]
                feat2 = features_with_labels[c2][0]
                n_samples = min(len(feat1), len(feat2))
                cka_score = CKA.linear_CKA(feat1[:n_samples], feat2[:n_samples])
                cka_matrix[i, j] = cka_matrix[j, i] = cka_score
    
    # Experiment 1B: Transfer matrix
    print("\nExperiment 1B: Cross-Encoding Transfer Analysis")
    transfer_matrix = np.zeros((n, n))
    for i, train_chart in enumerate(chart_types):
        for j, test_chart in enumerate(chart_types):
            train_feat, train_labels = features_with_labels[train_chart]
            test_feat, test_labels = features_with_labels[test_chart]
            
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_feat)
            test_scaled = scaler.transform(test_feat)
            
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(train_scaled, train_labels)
            acc = clf.score(test_scaled, test_labels)
            transfer_matrix[i, j] = acc
    
    # Experiment 1C: Intrinsic dimensionality
    print("\nExperiment 1C: Intrinsic Dimensionality Analysis (PCA)")
    dimensionality_results = compute_intrinsic_dimensionality(
        features_with_labels, chart_types, variance_threshold=0.90
    )
    
    # Experiment 1D: UMAP visualization
    print("\nExperiment 1D: Feature Space Visualization (UMAP)")
    analysis_dir = Path(output_dir) / dataset_name
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    create_umap_visualizations(features_with_labels, chart_types, dataset_name, analysis_dir)
    
    # Save results
    save_all_results(cka_matrix, transfer_matrix, dimensionality_results,
                    chart_types, dataset_name, analysis_dir)
    
    print(f"\nAnalysis complete. Results saved to: {analysis_dir}")
    return cka_matrix, transfer_matrix, dimensionality_results

def save_all_results(cka_matrix, transfer_matrix, dimensionality_results,
                    chart_types, dataset_name, output_dir):
    """Save all experimental results"""
    
    # Save CKA heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cka_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
                xticklabels=chart_types, yticklabels=chart_types,
                vmin=0, vmax=1, square=True, cbar_kws={'label': 'CKA Similarity'})
    plt.title(f'Experiment 1A: CKA Similarity Matrix - {dataset_name}', 
             fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'exp1a_cka_heatmap_{dataset_name}.png', dpi=300)
    plt.close()
    
    # Save transfer matrix heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(transfer_matrix * 100, annot=True, fmt='.1f', cmap='YlGnBu',
                xticklabels=chart_types, yticklabels=chart_types,
                vmin=0, vmax=100, square=True, cbar_kws={'label': 'Accuracy (%)'})
    plt.title(f'Experiment 1B: Transfer Matrix - {dataset_name}', 
             fontweight='bold', fontsize=14)
    plt.xlabel('Test Chart Type', fontsize=12)
    plt.ylabel('Train Chart Type', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / f'exp1b_transfer_matrix_{dataset_name}.png', dpi=300)
    plt.close()
    
    # Save dimensionality table
    dim_df = pd.DataFrame({
        'Chart Type': chart_types,
        'Components (90% var)': [dimensionality_results[ct]['n_components_90'] for ct in chart_types],
        'Actual Variance': [f"{dimensionality_results[ct]['variance_90']:.1%}" for ct in chart_types]
    })
    dim_df.to_csv(output_dir / f'exp1c_dimensionality_{dataset_name}.csv', index=False)
    
    # Save dimensionality plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(chart_types, 
                   [dimensionality_results[ct]['n_components_90'] for ct in chart_types],
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.xlabel('Chart Type', fontsize=12)
    plt.ylabel('Number of Components (90% variance)', fontsize=12)
    plt.title(f'Experiment 1C: Intrinsic Dimensionality - {dataset_name}', 
             fontweight='bold', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'exp1c_dimensionality_plot_{dataset_name}.png', dpi=300)
    plt.close()
    
    # Generate comprehensive report
    with open(output_dir / f'comprehensive_report_{dataset_name}.txt', 'w', encoding='utf-8') as f:
        f.write(f"Week 1-2 Comprehensive Analysis Report\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write("="*70 + "\n\n")
        
        # Experiment 1A
        f.write("Experiment 1A: CKA Similarity Analysis\n")
        f.write("-"*70 + "\n")
        n = len(chart_types)
        avg_cka = np.mean(cka_matrix[np.triu_indices(n, k=1)])
        f.write(f"Average CKA similarity: {avg_cka:.4f}\n\n")
        
        f.write("Pairwise similarities:\n")
        for i, c1 in enumerate(chart_types):
            for j, c2 in enumerate(chart_types):
                if i < j:
                    f.write(f"  {c1} - {c2}: {cka_matrix[i, j]:.4f}\n")
        
        if avg_cka > 0.8:
            f.write("\n✓ Different encodings create highly similar feature spaces\n")
        elif avg_cka > 0.5:
            f.write("\n~ Different encodings create moderately similar feature spaces\n")
        else:
            f.write("\n✗ Different encodings create significantly different feature spaces\n")
        
        # Experiment 1B
        f.write("\n\nExperiment 1B: Cross-Encoding Transfer Analysis\n")
        f.write("-"*70 + "\n")
        diag = np.mean(np.diag(transfer_matrix))
        off_diag = transfer_matrix[~np.eye(n, dtype=bool)]
        gap = diag - np.mean(off_diag)
        
        f.write(f"Same-type accuracy: {diag:.4f}\n")
        f.write(f"Cross-type accuracy: {np.mean(off_diag):.4f}\n")
        f.write(f"Transfer gap: {gap:.4f}\n\n")
        
        if gap < 0.1:
            f.write("✓ Features are highly transferable, encoding has minimal impact\n")
        elif gap < 0.2:
            f.write("~ Features are moderately transferable\n")
        else:
            f.write("✗ Features are difficult to transfer, encoding choice is important\n")
        
        # Experiment 1C
        f.write("\n\nExperiment 1C: Intrinsic Dimensionality Analysis\n")
        f.write("-"*70 + "\n")
        f.write("Number of principal components needed to explain 90% variance:\n\n")
        
        for ct in chart_types:
            n_comp = dimensionality_results[ct]['n_components_90']
            var = dimensionality_results[ct]['variance_90']
            f.write(f"  {ct:8s}: {n_comp:3d} components ({var:.1%} variance)\n")
        
        dims = [dimensionality_results[ct]['n_components_90'] for ct in chart_types]
        min_dim = min(dims)
        max_dim = max(dims)
        
        f.write(f"\nDimensionality range: {min_dim} - {max_dim}\n")
        
        if max_dim - min_dim < 10:
            f.write("→ All encodings have similar intrinsic dimensionality\n")
        else:
            min_chart = chart_types[dims.index(min_dim)]
            max_chart = chart_types[dims.index(max_dim)]
            f.write(f"→ {min_chart} has lowest dimensionality ({min_dim})\n")
            f.write(f"→ {max_chart} has highest dimensionality ({max_dim})\n")
        
        # Experiment 1D
        f.write("\n\nExperiment 1D: UMAP Visualization Analysis\n")
        f.write("-"*70 + "\n")
        f.write("Generated two visualizations:\n")
        f.write(f"  1. Colored by true class: umap_by_class_{dataset_name}.png\n")
        f.write(f"  2. Colored by chart type: umap_by_chart_{dataset_name}.png\n\n")
        f.write("View charts to analyze:\n")
        f.write("  - Class separation quality (chart 1)\n")
        f.write("  - Whether different encodings form separate clusters (chart 2)\n")
        
        # Summary
        f.write("\n\n" + "="*70 + "\n")
        f.write("RQ1 Answer Summary\n")
        f.write("="*70 + "\n\n")
        f.write("Do different visual encodings create different feature spaces?\n\n")
        
        if avg_cka > 0.7 and gap < 0.15:
            f.write("Answer: No\n")
            f.write("Evidence:\n")
            f.write(f"  • High CKA similarity ({avg_cka:.3f}) indicates similar representations\n")
            f.write(f"  • Low transfer gap ({gap:.3f}) indicates transferable features\n")
            f.write(f"  • Similar intrinsic dimensionality (range: {min_dim}-{max_dim})\n")
            f.write("\nPractical implication: Chart type choice has minimal impact on final performance\n")
        else:
            f.write("Answer: Yes\n")
            f.write("Evidence:\n")
            f.write(f"  • CKA similarity ({avg_cka:.3f}) indicates representation differences\n")
            f.write(f"  • Transfer gap ({gap:.3f}) indicates difficulty in cross-type transfer\n")
            if max_dim - min_dim >= 10:
                f.write(f"  • Significant dimensionality differences ({min_dim} vs {max_dim})\n")
            f.write("\nPractical implication: Chart type choice significantly affects model behavior\n")

# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Complete Week 1-2 Experimental Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:

# Full pipeline with chart generation
python week1_2_pipeline.py --ucr_data_dir /path/to/UCRArchive2018 --dataset ECG5000 --epochs 30

# Use existing charts
python week1_2_pipeline.py --existing_charts_dir /path/to/charts --ucr_data_dir /path/to/UCRArchive2018 --dataset ECG5000 --epochs 30

# Skip training (use saved models)
python week1_2_pipeline.py --ucr_data_dir /path/to/UCRArchive2018 --dataset ECG5000 --skip_training

# Quick test (fewer epochs, fewer chart types)
python week1_2_pipeline.py --ucr_data_dir /path/to/UCRArchive2018 --dataset ECG5000 --epochs 10 --chart_types line area
        """
    )
    
    parser.add_argument('--ucr_data_dir', type=str, required=True,
                       help='UCR dataset root directory')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., ECG5000)')
    parser.add_argument('--existing_charts_dir', type=str, default=None,
                       help='Directory with pre-generated charts (skip chart generation if provided)')
    parser.add_argument('--output_root', type=str, default='./week1_2_output',
                       help='Output root directory')
    parser.add_argument('--chart_types', nargs='+', 
                       default=['line', 'area', 'scatter', 'bar'],
                       help='Chart types to analyze')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Training epochs')
    parser.add_argument('--input_size', type=int, default=None,
                       help='Model input size (auto-detect if not specified)')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training (load existing models)')
    
    args = parser.parse_args()
    
    output_root = Path(args.output_root)
    chart_dir = Path(args.existing_charts_dir) if args.existing_charts_dir else output_root / 'charts'
    model_dir = output_root / 'models'
    analysis_dir = output_root / 'analysis'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    start_time = time.time()
    
    print("\n" + "="*60)
    print(f"Week 1-2 Complete Experimental Pipeline")
    print(f"Dataset: {args.dataset}")
    print(f"Chart types: {args.chart_types}")
    print("="*60)
    
    try:
        # Stage 1: Generate or load charts
        if args.existing_charts_dir:
            print("\n[Stage 1/3] Using existing charts")
            dataset_path = Path(args.ucr_data_dir) / args.dataset
            _, train_labels = load_ucr_data(dataset_path / f"{args.dataset}_TRAIN.tsv")
            _, test_labels = load_ucr_data(dataset_path / f"{args.dataset}_TEST.tsv")
            train_labels = normalize_labels(train_labels)
            test_labels = normalize_labels(test_labels)
        else:
            print("\n[Stage 1/3] Generating charts")
            train_labels, test_labels = generate_charts_for_dataset(
                args.dataset, args.ucr_data_dir, chart_dir, args.chart_types
            )
        
        # Auto-detect input size if not specified
        if args.input_size is None:
            sample_chart = list((chart_dir / args.dataset).glob('*_charts_color_with_label/train/*.png'))[0]
            sample_img = Image.open(sample_chart)
            args.input_size = sample_img.size[0]
            print(f"Auto-detected input size: {args.input_size}")
        
        # Stage 2: Train models
        if not args.skip_training:
            print("\n[Stage 2/3] Training models")
            models = train_all_models(
                args.dataset, chart_dir, train_labels, test_labels,
                args.chart_types, model_dir, args.epochs, device, args.input_size
            )
        else:
            print("\n[Stage 2/3] Loading existing models")
            models = {}
            
            transform = transforms.Compose([
                transforms.Resize((args.input_size, args.input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            num_classes = len(set(test_labels))
            
            for chart_type in args.chart_types:
                model_path = model_dir / args.dataset / f"{chart_type}_model.pth"
                
                if not model_path.exists():
                    print(f"  Model not found: {model_path}")
                    continue
                
                model = ImprovedDeep2DCNN(num_classes, args.input_size).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                
                chart_subdir = chart_dir / args.dataset / f"{chart_type}_charts_color_with_label"
                test_dir = chart_subdir / "test"
                test_dataset = SimpleDataset(test_dir, test_labels, transform)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
                
                models[chart_type] = (model, test_loader)
                print(f"  Loaded {chart_type} model")
        
        # Stage 3: Representation analysis
        print("\n[Stage 3/3] Representation analysis")
        cka_matrix, transfer_matrix, dimensionality_results = run_analysis(
            models, args.dataset, analysis_dir, list(models.keys()), device
        )
        
        # Summary
        elapsed = time.time() - start_time
        print("\n" + "="*60)
        print("Experiment complete!")
        print("="*60)
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"\nResults location:")
        print(f"  Charts: {chart_dir / args.dataset}")
        print(f"  Models: {model_dir / args.dataset}")
        print(f"  Analysis: {analysis_dir / args.dataset}")
        
        avg_cka = np.mean(cka_matrix[np.triu_indices(len(models), k=1)])
        diag = np.mean(np.diag(transfer_matrix))
        off_diag = transfer_matrix[~np.eye(len(models), dtype=bool)]
        gap = diag - np.mean(off_diag)
        
        print(f"\nKey findings:")
        print(f"  Average CKA similarity: {avg_cka:.3f}")
        print(f"  Transfer gap: {gap:.3f}")
        
        if avg_cka > 0.7 and gap < 0.15:
            print(f"  Conclusion: Different encodings learn similar features, choice has minimal impact")
        else:
            print(f"  Conclusion: Different encodings learn different features, choice is important")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
