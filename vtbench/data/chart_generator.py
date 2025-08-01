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


# Enhanced chart generation classes and functions

class GlobalYRangeCalculator:
    """Calculate and manage global Y-axis range"""
    
    @staticmethod
    def calculate_global_y_range(time_series_list: list, 
                               margin_ratio: float = 0.05) -> tuple:
        """Calculate global Y-axis range
        
        Args:
            time_series_list: List of time series data
            margin_ratio: Margin ratio to add padding to min/max values
            
        Returns:
            (global_min, global_max): Tuple of global minimum and maximum values
        """
        if len(time_series_list) == 0:
            return (0.0, 1.0)
        
        all_mins = []
        all_maxs = []
        
        for ts in time_series_list:
            if len(ts) > 0:
                # Clean data: handle NaN, inf and other outliers
                ts_clean = np.nan_to_num(ts.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
                all_mins.append(np.min(ts_clean))
                all_maxs.append(np.max(ts_clean))
        
        if not all_mins:
            return (0.0, 1.0)
        
        global_min = min(all_mins)
        global_max = max(all_maxs)
        
        # Add margin
        data_range = global_max - global_min
        if data_range > 0:
            margin = data_range * margin_ratio
            global_min -= margin
            global_max += margin
        else:
            # If data range is 0, add fixed margin
            global_min -= 0.1
            global_max += 0.1
        
        return (global_min, global_max)


class EnhancedImageGenerator:
    """Enhanced image generator - supports multiple chart types, color modes and label modes"""
    
    def __init__(self, height: int, width: int, 
                 color_mode: str = 'color', 
                 label_mode: str = 'with_label',
                 content_ratio: float = 0.95, 
                 global_y_range: tuple = None):
        """Initialize image generator
        
        Args:
            height: Image height
            width: Image width
            color_mode: Color mode ('color' or 'monochrome')
            label_mode: Label mode ('with_label' or 'without_label')
            content_ratio: Content area ratio (0-1)
            global_y_range: Global Y-axis range (min, max) or None
        """
        self.height = height
        self.width = width
        self.color_mode = color_mode
        self.label_mode = label_mode
        self.content_ratio = content_ratio
        self.global_y_range = global_y_range
        
        # Calculate content area dimensions
        self.content_width = int(width * content_ratio)
        self.content_height = int(height * content_ratio)
        self.margin_x = (width - self.content_width) // 2
        self.margin_y = (height - self.content_height) // 2
    
    def generate_image(self, time_series: np.ndarray, chart_type: str = 'area', 
                      class_label: int = None) -> np.ndarray:
        """Generate high quality time series image
        
        Args:
            time_series: Time series data
            chart_type: Chart type ('area', 'line', 'scatter', 'bar')
            class_label: Class label (optional, not used in current version)
            
        Returns:
            Image as numpy array (C, H, W), values in range [0,1]
        """
        # Clean input data
        ts_clean = np.nan_to_num(time_series.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
        return self._create_matplotlib_chart(ts_clean, chart_type, class_label)
    
    def _create_matplotlib_chart(self, ts: np.ndarray, chart_type: str, 
                                class_label: int) -> np.ndarray:
        """Core method to create chart using matplotlib"""
        if len(ts) == 0:
            # Return blank image
            return np.ones((3, self.height, self.width), dtype=np.float32)
        
        # Create figure with size and DPI settings
        fig_width = self.width / 100
        fig_height = self.height / 100
        
        plt.figure(figsize=(fig_width, fig_height), dpi=100, facecolor='white')
        ax = plt.gca()
        
        x_data = np.arange(len(ts))
        
        # Draw based on chart type
        if chart_type == 'area':
            color = 'blue' if self.color_mode == 'color' else 'black'
            plt.fill_between(x_data, ts, color=color)
            
        elif chart_type == 'line':
            color = 'blue' if self.color_mode == 'color' else 'black'
            plt.plot(ts, color=color)
            
        elif chart_type == 'scatter':
            color = 'blue' if self.color_mode == 'color' else 'black'
            plt.scatter(x_data, ts, color=color)
            
        elif chart_type == 'bar':
            color = 'blue' if self.color_mode == 'color' else 'black'
            plt.bar(x_data, ts, color='none', edgecolor=color, width=1.0)
        
        # Set Y-axis range (if global range is specified)
        if self.global_y_range is not None:
            y_min, y_max = self.global_y_range
            plt.ylim(y_min, y_max)
        
        # Configure axes and labels
        self._configure_axes(chart_type)
        
        # Convert to image array
        image = self._convert_figure_to_array()
        plt.close()
        
        return image
    
    def _configure_axes(self, chart_type: str):
        """Configure axis display"""
        if self.label_mode == 'with_label':
            # With label mode: show title
            chart_titles = {
                'area': 'Area Chart',
                'line': 'Line Chart', 
                'scatter': 'Scatter Chart',
                'bar': 'Bar Chart'
            }
            title = chart_titles.get(chart_type, f'{chart_type.title()} Chart')
            plt.title(title)
            # Keep matplotlib default axis labels and ticks
            
        elif self.label_mode == 'without_label':
            # Without label mode: hide all axes and labels
            plt.axis('off')
    
    def _convert_figure_to_array(self) -> np.ndarray:
        """Convert matplotlib figure to numpy array"""
        # Save figure to memory buffer
        from io import BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                   pad_inches=0, facecolor='white', edgecolor='none')
        buf.seek(0)
        
        # Load image using PIL
        pil_img = Image.open(buf)
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        # Resize to target dimensions
        pil_img = pil_img.resize((self.width, self.height), Image.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(pil_img)
        buf.close()
        
        # Normalize to [0,1] range and convert to (C,H,W) format
        image = image_array.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)  # (H,W,C) -> (C,H,W)
        
        return image


def save_image_array(image: np.ndarray, output_path: str):
    """Save image array to file
    
    Args:
        image: Image array with shape (C,H,W), values in range [0,1]
        output_path: Output file path
    """
    # Convert to uint8 format
    image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    # Convert to PIL format (H,W,C)
    image_pil = Image.fromarray(image_uint8.transpose(1, 2, 0), 'RGB')
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Save image
    image_pil.save(output_path, 'PNG')


# Convenience functions

def create_time_series_chart(time_series: np.ndarray, 
                           output_path: str,
                           chart_type: str = 'area',
                           color_mode: str = 'color',
                           label_mode: str = 'with_label',
                           width: int = 256,
                           height: int = 256,
                           global_y_range: tuple = None) -> np.ndarray:
    """Convenience function to create a single time series chart
    
    Args:
        time_series: Time series data
        output_path: Output file path
        chart_type: Chart type ('area', 'line', 'scatter', 'bar')
        color_mode: Color mode ('color', 'monochrome')
        label_mode: Label mode ('with_label', 'without_label')
        width: Image width
        height: Image height
        global_y_range: Global Y-axis range (min, max) or None
        
    Returns:
        Generated image array
    """
    generator = EnhancedImageGenerator(
        height=height, 
        width=width, 
        color_mode=color_mode, 
        label_mode=label_mode,
        global_y_range=global_y_range
    )
    
    image = generator.generate_image(time_series, chart_type)
    save_image_array(image, output_path)
    
    return image


def create_multiple_charts(time_series_list: list,
                         output_dir: str,
                         chart_types: list = ['area'],
                         color_modes: list = ['color'],
                         label_modes: list = ['with_label'],
                         width: int = 256,
                         height: int = 256,
                         use_global_y_range: bool = True) -> list:
    """Batch create multiple time series charts
    
    Args:
        time_series_list: List of time series data
        output_dir: Output directory
        chart_types: List of chart types
        color_modes: List of color modes
        label_modes: List of label modes
        width: Image width
        height: Image height
        use_global_y_range: Whether to use global Y-axis range
        
    Returns:
        List of generated file paths
    """
    # Calculate global Y-axis range
    global_y_range = None
    if use_global_y_range:
        global_y_range = GlobalYRangeCalculator.calculate_global_y_range(time_series_list)
    
    created_files = []
    
    # Generate all combinations
    for i, ts in enumerate(time_series_list):
        for chart_type in chart_types:
            for color_mode in color_modes:
                for label_mode in label_modes:
                    filename = f"chart_{i}_{chart_type}_{color_mode}_{label_mode}.png"
                    output_path = os.path.join(output_dir, filename)
                    
                    create_time_series_chart(
                        time_series=ts,
                        output_path=output_path,
                        chart_type=chart_type,
                        color_mode=color_mode,
                        label_mode=label_mode,
                        width=width,
                        height=height,
                        global_y_range=global_y_range
                    )
                    
                    created_files.append(output_path)
    
    return created_files



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
