# chart_generator.py
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset


# ==============================
# Utilities & Helper Functions
# ==============================

def _sota_grid_dims(D: int) -> tuple[int, int]:
    """
    Grid rule:
      - (ℓ−1) × ℓ if D ≤ ℓ(ℓ−1)
      - ℓ × ℓ     if ℓ(ℓ−1) < D ≤ ℓ²
      - ℓ × (ℓ+1) if ℓ² < D ≤ ℓ(ℓ+1)
    where ℓ = ceil(sqrt(D)).
    """
    if D <= 0:
        return (1, 1)
    l = int(np.ceil(np.sqrt(D)))
    if D <= l * (l - 1):
        return (l - 1, l)
    elif D <= l * l:
        return (l, l)
    else:
        return (l, l + 1)


def _title_for(chart_type: str) -> str:
    return {
        'area': 'Area Chart',
        'line': 'Line Chart',
        'scatter': 'Scatter Chart',
        'bar': 'Bar Chart'
    }.get(chart_type, f'{chart_type.title()} Chart')


def _coerce_numeric_1d(ts) -> np.ndarray:
    """
    Return 1-D float32 array with NaN/inf replaced (0.0).
    Accepts (T,), (1,T), lists, and object arrays with mixed tokens (e.g., "v:1.23").
    """
    arr = np.asarray(ts, dtype=object)

    # Squeeze singleton dimension, e.g., (1, T) -> (T,)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.squeeze()

    # Fast cast path
    try:
        out = arr.astype(np.float32, copy=False)
    except Exception:
        # Fallback: parse token-by-token
        vals = []
        for x in arr.ravel().tolist():
            try:
                vals.append(float(str(x).split(":")[-1].strip()))
            except Exception:
                vals.append(0.0)
        out = np.asarray(vals, dtype=np.float32)

    # Replace NaN/inf
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure 1-D
    if out.ndim != 1:
        out = out.reshape(-1)
    return out


def _is_multivariate(ts: np.ndarray) -> bool:
    return isinstance(ts, np.ndarray) and ts.ndim == 2 and ts.shape[0] > 1


# ==========================
# Save helpers (dynamic size)
# ==========================

def _save_with_labels_dynamic(fig, path, dpi=100, pad=0.6):
    """Keep titles/ticks; use tight_layout to avoid clipping."""
    plt.tight_layout(pad=pad)
    fig.savefig(path, dpi=dpi, bbox_inches=None, pad_inches=0, facecolor='white')
    plt.close(fig)


def _save_without_labels_dynamic(fig, path, dpi=100):
    """No axes/ticks; fill canvas with data (no margins)."""
    ax = fig.gca()
    ax.set_position([0, 0, 1, 1])
    fig.savefig(path, dpi=dpi, bbox_inches=None, pad_inches=0, facecolor='white')
    plt.close(fig)


# ==========================
# Univariate Chart Creators
# ==========================

def _style_small_axes_with_labels(ax, chart_type):
    ax.set_title(_title_for(chart_type), fontsize=8, pad=2)
    ax.tick_params(labelsize=6, width=0.6)
    for s in ax.spines.values():
        s.set_linewidth(0.6)


def create_line_chart(ts, chart_path, color_mode, label_mode,
                      global_y_range=None):
    v = _coerce_numeric_1d(ts)
    fig = plt.figure()  # dynamic default size (e.g., 640x480 @ dpi=100)
    ax = fig.gca()
    color = 'blue' if color_mode == 'color' else 'black'
    ax.plot(v, color=color, linewidth=1.0)
    if global_y_range is not None:
        ax.set_ylim(*global_y_range)
    if label_mode == 'without_label':
        ax.axis('off')
        _save_without_labels_dynamic(fig, chart_path)
    else:
        _style_small_axes_with_labels(ax, 'line')
        _save_with_labels_dynamic(fig, chart_path)


def create_area_chart(ts, chart_path, color_mode, label_mode,
                      global_y_range=None):
    v = _coerce_numeric_1d(ts)
    fig = plt.figure()
    ax = fig.gca()
    color = 'blue' if color_mode == 'color' else 'black'
    x = np.arange(len(v))
    ax.fill_between(x, v, color=color)
    if global_y_range is not None:
        ax.set_ylim(*global_y_range)
    if label_mode == 'without_label':
        ax.axis('off')
        _save_without_labels_dynamic(fig, chart_path)
    else:
        _style_small_axes_with_labels(ax, 'area')
        _save_with_labels_dynamic(fig, chart_path)


def create_bar_chart(ts, chart_path, bar_mode, color_mode, label_mode,
                     global_y_range=None):
    v = _coerce_numeric_1d(ts)
    fig = plt.figure()
    ax = fig.gca()
    color = 'blue' if color_mode == 'color' else 'black'
    if bar_mode is None:
        bar_mode = 'border'
    x = np.arange(len(v))
    ax.bar(x, v, color='none', edgecolor=color, width=1.0)
    if global_y_range is not None:
        ax.set_ylim(*global_y_range)
    if label_mode == 'without_label':
        ax.axis('off')
        _save_without_labels_dynamic(fig, chart_path)
    else:
        _style_small_axes_with_labels(ax, 'bar')
        _save_with_labels_dynamic(fig, chart_path)


def create_scatter_chart(ts, chart_path, scatter_mode, color_mode, label_mode,
                         global_y_range=None):
    v = _coerce_numeric_1d(ts)
    fig = plt.figure()
    ax = fig.gca()
    color = 'blue' if color_mode == 'color' else 'black'
    x = np.arange(len(v))
    if scatter_mode == 'join':
        ax.plot(v, color=color, linewidth=0.9)
    ax.scatter(x, v, color=color, s=6)
    if global_y_range is not None:
        ax.set_ylim(*global_y_range)
    if label_mode == 'without_label':
        ax.axis('off')
        _save_without_labels_dynamic(fig, chart_path)
    else:
        _style_small_axes_with_labels(ax, 'scatter')
        _save_with_labels_dynamic(fig, chart_path)


# ======================================
# Multivariate Grid (Exact Pixel Output)
# ======================================

def create_multivariate_grid_chart(
    ts_matrix,
    chart_path,
    chart_type,
    color_mode,
    label_mode,
    *,
    cell_px: int = 64,   # 64x64 per cell
    dpi: int = 100,
    linewidth: float = 1.5,
    scatter_s: float = 4.0,
    global_y_range=None  # dataset-level global Y range for multi
):
    """
    Save an image with pixel size: (cols*cell_px) × (rows*cell_px).
    """
    # Robust numeric coercion for 2D
    try:
        ts_obj = np.asarray(ts_matrix, dtype=object)
    except Exception:
        ts_obj = np.array(ts_matrix, dtype=object)

    if ts_obj.ndim != 2:
        ts_obj = ts_obj.reshape(1, -1)

    # Coerce each row to numeric 1D
    rows_list = []
    for i in range(ts_obj.shape[0]):
        rows_list.append(_coerce_numeric_1d(ts_obj[i]))
    # Pad/truncate rows to the same length if needed
    max_T = max(len(r) for r in rows_list)
    ts = np.zeros((len(rows_list), max_T), dtype=np.float32)
    for i, r in enumerate(rows_list):
        if len(r) == max_T:
            ts[i] = r
        elif len(r) < max_T:
            ts[i, :len(r)] = r
        else:
            ts[i] = r[:max_T]

    V, T = ts.shape

    # If (1, T) slipped in, reroute to univariate creator
    if V == 1:
        if chart_type == 'area':
            create_area_chart(ts[0], chart_path, color_mode, label_mode, global_y_range)
        elif chart_type == 'bar':
            create_bar_chart(ts[0], chart_path, 'border', color_mode, label_mode, global_y_range)
        elif chart_type == 'line':
            create_line_chart(ts[0], chart_path, color_mode, label_mode, global_y_range)
        elif chart_type == 'scatter':
            create_scatter_chart(ts[0], chart_path, 'plain', color_mode, label_mode, global_y_range)
        return

    rows, cols = _sota_grid_dims(V)

    # Global Y range for multivariate
    if global_y_range is not None:
        y_min, y_max = global_y_range
    else:
        y_min = float(np.min(ts))
        y_max = float(np.max(ts))
        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
            y_min, y_max = y_min - 0.1, y_max + 0.1

    fig_w_in = (cols * cell_px) / dpi
    fig_h_in = (rows * cell_px) / dpi

    plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi, facecolor='white')
    color = ('blue' if color_mode == 'color' else 'black')
    chart_title = _title_for(chart_type)

    for i in range(V):
        ax = plt.subplot(rows, cols, i + 1)
        v = ts[i]

        if chart_type == 'line':
            ax.plot(v, color=color, linewidth=linewidth)
        elif chart_type == 'area':
            ax.fill_between(np.arange(T), v, color=color)
        elif chart_type == 'scatter':
            ax.scatter(np.arange(T), v, c=color, s=scatter_s)
        elif chart_type == 'bar':
            if color_mode == 'monochrome':
                ax.bar(np.arange(T), v, color='none', edgecolor='black', width=1.0)
            else:
                ax.bar(np.arange(T), v, color='blue', edgecolor='blue', width=1.0)
        else:
            ax.plot(v, color=color, linewidth=linewidth)

        ax.set_ylim(y_min, y_max)
        if label_mode == 'without_label':
            ax.axis('off')
        else:
            ax.set_title(chart_title, fontsize=8)
            ax.tick_params(labelsize=6)

    # Fill remaining cells (blank or titled stubs)
    for j in range(V + 1, rows * cols + 1):
        ax = plt.subplot(rows, cols, j)
        if label_mode == 'without_label':
            ax.axis('off')
        else:
            ax.set_title(chart_title, fontsize=8)
            ax.tick_params(labelsize=6)
            ax.set_xticks([]); ax.set_yticks([])

    # Keep a little breathing room when labels are on
    if label_mode == 'without_label':
        plt.tight_layout(pad=0.0)
    else:
        plt.tight_layout(pad=0.8)

    plt.savefig(chart_path, dpi=dpi, bbox_inches=None, pad_inches=0, facecolor='white')
    plt.close()


# ============================
# Global Y-Range Calculators
# ============================

class GlobalYRangeCalculator:
    @staticmethod
    def calculate_global_y_range_univariate(time_series_list: list, margin_ratio: float = 0.05):
        vals_min = +np.inf
        vals_max = -np.inf
        for ts in time_series_list:
            arr = _coerce_numeric_1d(ts)
            if arr.size == 0:
                continue
            vals_min = min(vals_min, float(np.min(arr)))
            vals_max = max(vals_max, float(np.max(arr)))
        if not np.isfinite(vals_min) or not np.isfinite(vals_max):
            return None
        rng = vals_max - vals_min
        if rng <= 0:
            return (vals_min - 0.1, vals_max + 0.1)
        m = margin_ratio * rng
        return (vals_min - m, vals_max + m)

    @staticmethod
    def calculate_global_y_range_multivariate(time_series_list: list, margin_ratio: float = 0.05):
        vals_min = +np.inf
        vals_max = -np.inf
        for ts in time_series_list:
            arr_obj = np.asarray(ts, dtype=object)
            if arr_obj.ndim == 2:
                # Coerce each row to numeric and flatten
                rows = []
                for i in range(arr_obj.shape[0]):
                    rows.append(_coerce_numeric_1d(arr_obj[i]))
                if len(rows) == 0:
                    continue
                max_T = max(len(r) for r in rows)
                if max_T == 0:
                    continue
                arr = np.zeros((len(rows), max_T), dtype=np.float32)
                for i, r in enumerate(rows):
                    if len(r) == max_T:
                        arr[i] = r
                    elif len(r) < max_T:
                        arr[i, :len(r)] = r
                    else:
                        arr[i] = r[:max_T]
                arr = arr.reshape(-1)
            else:
                arr = _coerce_numeric_1d(arr_obj)

            if arr.size == 0:
                continue

            vals_min = min(vals_min, float(np.min(arr)))
            vals_max = max(vals_max, float(np.max(arr)))

        if not np.isfinite(vals_min) or not np.isfinite(vals_max):
            return None
        rng = vals_max - vals_min
        if rng <= 0:
            return (vals_min - 0.1, vals_max + 0.1)
        m = margin_ratio * rng
        return (vals_min - m, vals_max + m)


# ==========================
# Image Dataset (PNG files)
# ==========================

class TimeSeriesImageDataset(Dataset):
    """
    Generates chart images once (if requested) and loads them for training.
    Let your training transforms resize to 128x128 later.
    Univariate -> dynamic matplotlib default size
    Multivariate -> exact (cols*cell_px) x (rows*cell_px) grid
    """

    def __init__(self, time_series_data, labels, dataset_name, split,
                 chart_type='area', color_mode='color', label_mode='with_label',
                 scatter_mode='plain', bar_mode='border', transform=None,
                 generate_images=False, overwrite_existing=False,
                 global_indices=None, multivariate_mode='subplots'):
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
        self.multivariate_mode = multivariate_mode

        self.base_dir = f"chart_images/{self.dataset_name}_images"
        self._setup_chart_dir()

        # Dataset-level global Y ranges (computed only when generating)
        self.uni_global_y = None
        self.multi_global_y = None
        if self.generate_images:
            self.uni_global_y = GlobalYRangeCalculator.calculate_global_y_range_univariate(self.time_series_data)
            self.multi_global_y = GlobalYRangeCalculator.calculate_global_y_range_multivariate(self.time_series_data)

        if self.generate_images:
            self._generate_charts_if_needed()

    def _setup_chart_dir(self):
        if self.chart_type == 'area':
            self.chart_dir = f"{self.base_dir}/area_charts_{self.color_mode}_{self.label_mode}/{self.split}"
        elif self.chart_type == 'bar':
            bm = self.bar_mode or 'border'
            self.chart_dir = f"{self.base_dir}/bar_charts_{bm}_{self.color_mode}_{self.label_mode}/{self.split}"
        elif self.chart_type == 'line':
            self.chart_dir = f"{self.base_dir}/line_charts_{self.color_mode}_{self.label_mode}/{self.split}"
        elif self.chart_type == 'scatter':
            sm = self.scatter_mode or 'plain'
            self.chart_dir = f"{self.base_dir}/scatter_charts_{sm}_{self.color_mode}_{self.label_mode}/{self.split}"
        else:
            raise ValueError(f"Unsupported chart type: {self.chart_type}")
        os.makedirs(self.chart_dir, exist_ok=True)

    def _get_image_filename(self, idx):
        actual_idx = self.global_indices[idx] if self.global_indices is not None else idx
        prefix = {
            'area': 'area_chart',
            'bar': 'bar_chart',
            'line': 'line_chart',
            'scatter': 'scatter_chart'
        }[self.chart_type]
        if self.chart_type == 'bar':
            bm = self.bar_mode or 'border'
            return f"{prefix}_{bm}_{self.color_mode}_{self.label_mode}_{actual_idx}.png"
        elif self.chart_type == 'scatter':
            sm = self.scatter_mode or 'plain'
            return f"{prefix}_{sm}_{self.color_mode}_{self.label_mode}_{actual_idx}.png"
        else:
            return f"{prefix}_{self.color_mode}_{self.label_mode}_{actual_idx}.png"

    def _generate_charts_if_needed(self):
        for local_idx, ts in enumerate(self.time_series_data):
            chart_path = os.path.join(self.chart_dir, self._get_image_filename(local_idx))
            if os.path.exists(chart_path) and not self.overwrite_existing:
                continue

            ts_arr = np.asarray(ts)
            # Treat (1, T) as univariate
            if ts_arr.ndim == 2 and ts_arr.shape[0] == 1:
                ts_arr = ts_arr.squeeze(0)

            if _is_multivariate(ts_arr):
                # Multivariate → grid with dataset-level global Y range
                create_multivariate_grid_chart(
                    ts_matrix=ts_arr,
                    chart_path=chart_path,
                    chart_type=self.chart_type,
                    color_mode=self.color_mode,
                    label_mode=self.label_mode,
                    cell_px=64,
                    global_y_range=self.multi_global_y
                )
            else:
                # Univariate → dynamic size; model will resize later
                gy = self.uni_global_y
                if self.chart_type == 'area':
                    create_area_chart(ts_arr, chart_path, self.color_mode, self.label_mode,
                                      global_y_range=gy)
                elif self.chart_type == 'bar':
                    create_bar_chart(ts_arr, chart_path, self.bar_mode, self.color_mode, self.label_mode,
                                     global_y_range=gy)
                elif self.chart_type == 'line':
                    create_line_chart(ts_arr, chart_path, self.color_mode, self.label_mode,
                                      global_y_range=gy)
                elif self.chart_type == 'scatter':
                    create_scatter_chart(ts_arr, chart_path, self.scatter_mode, self.color_mode, self.label_mode,
                                         global_y_range=gy)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        chart_path = os.path.join(self.chart_dir, self._get_image_filename(idx))
        img = Image.open(chart_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


# ==========================
# Numerical-only Dataset
# ==========================

class NumericalDataset(Dataset):
    def __init__(self, numerical_data, labels):
        self.numerical_data = torch.tensor(numerical_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.numerical_data[idx], self.labels[idx]


# ==========================
# Quick display helper
# ==========================

def display_chart(image_path):
    img = Image.open(image_path)
    img.show()
