import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset


######################################################### COLOR AND WITH LABELLING AND WITH BARCHART FILL ############################################################
# def create_area_chart_mc(time_series, save_path):
#     plt.figure()
#     plt.fill_between(range(len(time_series)), time_series, color="skyblue", alpha=0.4)
#     plt.plot(time_series, color="Slateblue", alpha=0.6)
#     plt.title('Area Chart')
#     plt.xlabel('Time')
#     plt.ylabel('Value')
#     plt.savefig(save_path)
#     plt.close()

# def create_bar_chart_mc(time_series, save_path):
#     plt.figure()
#     plt.bar(range(len(time_series)), time_series, color="skyblue", edgecolor="black", width=1.0)
#     plt.title('Bar Chart')
#     plt.xlabel('Time')
#     plt.ylabel('Value')
#     plt.savefig(save_path)
#     plt.close()

######################################################### COLOR AND WITHOUT LABELLING AND WITH BARCHART FILL ############################################################
# def create_area_chart_mc(time_series, save_path):
#     plt.figure()
#     plt.fill_between(range(len(time_series)), time_series, color="skyblue", alpha=0.4)
#     plt.plot(time_series, color="Slateblue", alpha=0.6)
#     plt.savefig(save_path)
#     plt.close()

# def create_bar_chart_mc(time_series, save_path):
#     plt.figure()
#     plt.bar(range(len(time_series)), time_series, color="skyblue", edgecolor="black", width=1.0)
#     plt.savefig(save_path)
#     plt.close()

######################################################### MONOCHROME AND WITH LABELLING AND WITH BARCHART AND AREA FILL - 29, 30, , 45 ############################################################
def create_area_chart_mc(time_series, save_path):
    plt.figure()
    plt.fill_between(range(len(time_series)), time_series, color="black", alpha=0.4)
    plt.plot(time_series, color="black", alpha=0.6)
    plt.title('Line Chart')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0) 
    plt.close()

def create_line_chart_mc(time_series, save_path):
    plt.figure()
    plt.plot(time_series, color="black", alpha=0.6)
    plt.title('Line Chart')
    plt.xlabel('Time')
    plt.ylabel('Value') 
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0) 
    plt.close()

def create_scatter_chart_mc(time_series, save_path):
    plt.figure()
    plt.scatter(range(len(time_series)), time_series, color="black")
    plt.title('Scatterplot')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0) 
    plt.close()

def create_bar_chart_mc(time_series, save_path):
    plt.figure()
    plt.bar(range(len(time_series)), time_series, color="black", edgecolor="black", width=1.0)
    plt.title('Bar Chart')
    plt.xlabel('Time')
    plt.ylabel('Value')  
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)  # Save the figure
    plt.close()

########################################################### MONOCHROME AND WITHOUT LABELLING AND WITH BARCHART FILL ###########################################################
# def create_area_chart_mc(time_series, save_path):
#     plt.figure()
#     plt.plot(time_series, color="black", alpha=0.6)
#     plt.axis('off') 
#     plt.savefig(save_path, bbox_inches='tight', pad_inches=0) 
#     plt.close()

# def create_bar_chart_mc(time_series, save_path):
#     plt.figure()
#     plt.bar(range(len(time_series)), time_series, color="black", edgecolor="black", width=1.0)
#     plt.axis('off')  
#     plt.savefig(save_path, bbox_inches='tight', pad_inches=0)  
#     plt.close()



class TimeSeriesImageDatasetMC(Dataset):
    def __init__(self, time_series_data, labels, split, transform=None, chart_type='area'):
        self.time_series_data = time_series_data
        self.labels = labels
        self.split = split
        self.transform = transform
        self.chart_type = chart_type
        self.area_chart_dir_mc = f'data/29area_charts_labelmono/{split}'
        self.line_chart_dir_mc = f'data/30line_charts_labelmono/{split}'
        self.scatter_chart_dir_mc = f'data/37scatter_charts_labelmono/{split}'
        self.bar_chart_dir_mc = f'data/45bar_charts_labelmono/{split}'
        os.makedirs(self.area_chart_dir_mc, exist_ok=True)
        os.makedirs(self.bar_chart_dir_mc, exist_ok=True)
        os.makedirs(self.scatter_chart_dir_mc, exist_ok=True)
        os.makedirs(self.line_chart_dir_mc, exist_ok=True)
        self._generate_charts()

    def _generate_charts(self):
        for i, ts in enumerate(self.time_series_data):
            area_path = os.path.join(self.area_chart_dir_mc, f'area_chart_{i}.png')
            bar_path = os.path.join(self.bar_chart_dir_mc, f'bar_chart_{i}.png')
            line_path = os.path.join(self.line_chart_dir_mc, f'line_chart_{i}.png')
            scatter_path = os.path.join(self.scatter_chart_dir_mc, f'scatter_chart_{i}.png')
            if not os.path.exists(area_path) and self.chart_type == 'area':
                create_area_chart_mc(ts, area_path)
            if not os.path.exists(bar_path) and self.chart_type == 'bar':
                create_bar_chart_mc(ts, bar_path)
            if not os.path.exists(line_path) and self.chart_type == 'line':
                create_line_chart_mc(ts, line_path)
            if not os.path.exists(scatter_path) and self.chart_type == 'scatter':
                create_scatter_chart_mc(ts, scatter_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.chart_type == 'area':
            img_path = os.path.join(self.area_chart_dir_mc, f'area_chart_{idx}.png')
        elif self.chart_type == 'bar':
            img_path = os.path.join(self.bar_chart_dir_mc, f'bar_chart_{idx}.png')
        elif self.chart_type == 'line':
            img_path = os.path.join(self.line_chart_dir_mc, f'line_chart_{idx}.png')
        elif self.chart_type == 'scatter':
            img_path = os.path.join(self.scatter_chart_dir_mc, f'scatter_chart_{idx}.png')
        else:
            raise ValueError(f"Unsupported chart type: {self.chart_type}")
        
        img = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

def display_chart_mc(image_path):
    img = Image.open(image_path)
    img.show()

def accuracy_fn_mc(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc
