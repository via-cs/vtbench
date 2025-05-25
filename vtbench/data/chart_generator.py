import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Chart plotting functions

def create_area_chart(ts, chart_path, color_mode, label_mode):
    plt.figure()
    plt.fill_between(range(len(ts)), ts, color='blue' if color_mode == 'color' else 'black')
    if label_mode == 'with_label':
        plt.title('Area Chart')
    else:
        plt.axis('off')
    plt.savefig(chart_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def create_bar_chart(ts, chart_path, bar_mode, color_mode, label_mode):
    plt.figure()
    color = 'blue' if color_mode == 'color' else 'black'
    bar_mode == 'border'
    plt.bar(range(len(ts)), ts, color='none', edgecolor=color, width=1.0)
    if label_mode == 'with_label':
        plt.title('Bar Chart')
    else:
        plt.axis('off')
    plt.savefig(chart_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def create_line_chart(ts, chart_path, color_mode, label_mode):
    plt.figure()
    plt.plot(ts, color='blue' if color_mode == 'color' else 'black')
    if label_mode == 'with_label':
        plt.title('Line Chart')
    else:
        plt.axis('off')
    plt.savefig(chart_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def create_scatter_chart(ts, chart_path, scatter_mode, color_mode, label_mode):
    plt.figure()
    color = 'blue' if color_mode == 'color' else 'black'
    if scatter_mode == 'plain':
        plt.scatter(range(len(ts)), ts, color=color)
    elif scatter_mode == 'join':
        plt.plot(ts, color='skyblue')
        plt.scatter(range(len(ts)), ts, color=color)
    if label_mode == 'with_label':
        plt.title('Scatter Chart')
    else:
        plt.axis('off')
    plt.savefig(chart_path, bbox_inches='tight', pad_inches=0)
    plt.close()



# Image dataset class

class TimeSeriesImageDataset(Dataset):
    """
    PyTorch Dataset for loading chart images generated from time series data.
    Automatically generates the images if not found.
    """

    def __init__(self, time_series_data, labels, dataset_name, split,
             chart_type='area', color_mode='color', label_mode='with_label',
             scatter_mode='plain', bar_mode='border', transform=None,
             generate_images=False, overwrite_existing=False,
             global_indices=None):
    
        self.time_series_data = time_series_data
        self.labels = labels
        self.dataset_name = dataset_name
        self.split = split
        self.chart_type = chart_type
        self.color_mode = color_mode
        self.label_mode = label_mode
        self.scatter_mode = scatter_mode
        self.bar_mode = bar_mode
        self.transform = transform
        self.generate_images = generate_images
        self.overwrite_existing = overwrite_existing
        self.global_indices = global_indices if global_indices is not None else list(range(len(labels)))

        self.base_dir = f"chart_images/{self.dataset_name}_images"
        self._setup_chart_dir()
        if self.generate_images:
            self._generate_charts_if_needed()


    def _setup_chart_dir(self):
        if self.chart_type == 'area':
            self.chart_dir = f"{self.base_dir}/area_charts_{self.color_mode}_{self.label_mode}/{self.split}"
        elif self.chart_type == 'bar':
            bar_mode = self.bar_mode or 'border'
            self.chart_dir = f"{self.base_dir}/bar_charts_{bar_mode}_{self.color_mode}_{self.label_mode}/{self.split}"
        elif self.chart_type == 'line':
            self.chart_dir = f"{self.base_dir}/line_charts_{self.color_mode}_{self.label_mode}/{self.split}"
        elif self.chart_type == 'scatter':
            self.chart_dir = f"{self.base_dir}/scatter_charts_{self.scatter_mode}_{self.color_mode}_{self.label_mode}/{self.split}"
        else:
            raise ValueError(f"Unsupported chart type: {self.chart_type}")
        
        os.makedirs(self.chart_dir, exist_ok=True)

    def _generate_charts_if_needed(self):
        for local_idx, ts in enumerate(self.time_series_data):
            chart_path = os.path.join(self.chart_dir, self._get_image_filename(local_idx))  # NOT global_idx
            if not os.path.exists(chart_path):
                if self.chart_type == 'area':
                    create_area_chart(ts, chart_path, self.color_mode, self.label_mode)
                elif self.chart_type == 'bar':
                    create_bar_chart(ts, chart_path, self.bar_mode, self.color_mode, self.label_mode)
                elif self.chart_type == 'line':
                    create_line_chart(ts, chart_path, self.color_mode, self.label_mode)
                elif self.chart_type == 'scatter':
                    create_scatter_chart(ts, chart_path, self.scatter_mode, self.color_mode, self.label_mode)



    def _get_image_filename(self, idx):
        actual_idx = self.global_indices[idx] if self.global_indices is not None else idx
        prefix = {
            'area': 'area_chart',
            'bar': 'bar_chart',
            'line': 'line_chart',
            'scatter': 'scatter_chart'
        }[self.chart_type]

        if self.chart_type == 'bar':
            return f"{prefix}_{self.bar_mode}_{self.color_mode}_{self.label_mode}_{actual_idx}.png"
        elif self.chart_type == 'scatter':
            return f"{prefix}_{self.scatter_mode}_{self.color_mode}_{self.label_mode}_{actual_idx}.png"
        else:
            return f"{prefix}_{self.color_mode}_{self.label_mode}_{actual_idx}.png"



    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        chart_path = os.path.join(self.chart_dir, self._get_image_filename(idx))
        img = Image.open(chart_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label
    

# Numerical Dataset 
class NumericalDataset(Dataset):
    def __init__(self, numerical_data, labels):
        self.numerical_data = torch.tensor(numerical_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.numerical_data[idx], self.labels[idx]


def display_chart(image_path):
    img = Image.open(image_path)
    img.show()
