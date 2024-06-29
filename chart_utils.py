#### Utils file for binary classification model with ECG

import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

def create_area_chart(time_series, save_path):
    plt.figure()
    plt.fill_between(range(len(time_series)), time_series, color="skyblue", alpha=0.4)
    plt.plot(time_series, color="Slateblue", alpha=0.6)
    plt.title('Area Chart')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.savefig(save_path)
    plt.close()

def create_bar_chart(time_series, save_path):
    plt.figure()
    plt.bar(range(len(time_series)), time_series, color="skyblue", edgecolor="black", width=1.0)
    plt.title('Bar Chart')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.savefig(save_path)
    plt.close()

class TimeSeriesImageDataset(Dataset):
    def __init__(self, time_series_data, labels, transform=None):
        self.time_series_data = time_series_data
        self.labels = labels
        self.transform = transform
        self.area_chart_dir = 'area_charts'
        self.bar_chart_dir = 'bar_charts'
        os.makedirs(self.area_chart_dir, exist_ok=True)
        os.makedirs(self.bar_chart_dir, exist_ok=True)
        self._generate_charts()

    def _generate_charts(self):
        for i, ts in enumerate(self.time_series_data):
            area_path = os.path.join(self.area_chart_dir, f'area_chart_{i}.png')
            bar_path = os.path.join(self.bar_chart_dir, f'bar_chart_{i}.png')
            create_area_chart(ts, area_path)
            create_bar_chart(ts, bar_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        area_img = Image.open(os.path.join(self.area_chart_dir, f'area_chart_{idx}.png')).convert('RGB')
        bar_img = Image.open(os.path.join(self.bar_chart_dir, f'bar_chart_{idx}.png')).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            area_img = self.transform(area_img)
            bar_img = self.transform(bar_img)

        return area_img, bar_img, label

def display_chart(image_path):
    img = Image.open(image_path)
    img.show()

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc
