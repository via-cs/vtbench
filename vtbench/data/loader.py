# import numpy as np
# import torch
# from torch.utils.data import DataLoader, Subset
# from torchvision import transforms
# from sklearn.model_selection import StratifiedShuffleSplit
# from vtbench.data.chart_generator import TimeSeriesImageDataset, NumericalDataset
# from collections import Counter
# import os

# def read_ucr(filename):
#     """
#     Robust UCR reader.
#     Handles:
#       - headers starting with '@' or '#'
#       - label either at END or at START of the row
#       - trailing commas / spaces
#       - numeric labels like '1', '1.0', or 'label:1'
#     Returns:
#       X: np.ndarray of shape (N, T) float32
#       y: np.ndarray of shape (N,) int64 in 0..C-1
#     """
#     import numpy as np

#     def _is_int_like(s):
#         try:
#             f = float(s)
#             return np.isfinite(f) and abs(f - round(f)) < 1e-6
#         except Exception:
#             return False

#     data = []
#     raw_labels = []

#     with open(filename, "r") as f:
#         for raw in f:
#             line = raw.strip()
#             if not line or line.startswith("@") or line.startswith("#"):
#                 continue

#             # tolerate trailing commas
#             line = line.rstrip(",")
#             parts = [p.strip() for p in line.split(",") if p.strip() != ""]
#             if len(parts) < 2:
#                 continue

#             # Try label at END
#             last_tok = parts[-1].split(":")[-1].strip()
#             if _is_int_like(last_tok):
#                 label = int(round(float(last_tok)))
#                 feat_strs = parts[:-1]
#             else:
#                 # Try label at START
#                 first_tok = parts[0].split(":")[-1].strip()
#                 if _is_int_like(first_tok):
#                     label = int(round(float(first_tok)))
#                     feat_strs = parts[1:]
#                 else:
#                     # Fallback: search for a token containing 'label:'
#                     found = None
#                     for tok in reversed(parts):
#                         if "label" in tok.lower():
#                             cand = tok.split(":")[-1].strip()
#                             if _is_int_like(cand):
#                                 label = int(round(float(cand)))
#                                 found = tok
#                                 break
#                     if found is None:
#                         # Cannot find a clean label → skip row
#                         continue
#                     feat_strs = [t for t in parts if t is not found]

#             # Convert features safely
#             feats = []
#             for s in feat_strs:
#                 s_clean = s.split(":")[-1].strip()
#                 try:
#                     feats.append(float(s_clean))
#                 except Exception:
#                     feats.append(0.0)  # safe fallback

#             if len(feats) == 0:
#                 continue

#             data.append(feats)
#             raw_labels.append(label)

#     # If nothing parsed, bail with a clear error
#     if len(data) == 0:
#         raise ValueError(f"No valid rows parsed from: {filename}")

#     X = np.asarray(data, dtype=np.float32)
#     labels_in = np.asarray(raw_labels)

#     # --- Normalize labels to 0..C-1 (keep your special cases) ---
#     label_set = set(int(l) for l in labels_in.tolist())

#     if label_set == {0, 1}:
#         normalize = lambda l: int(l)
#     elif label_set == {1, 2}:
#         normalize = lambda l: 0 if int(l) == 1 else 1
#     elif label_set == {-1, 1}:
#         normalize = lambda l: 0 if int(l) == -1 else 1
#     else:
#         sorted_labels = sorted(label_set)
#         label_map = {lab: idx for idx, lab in enumerate(sorted_labels)}
#         normalize = lambda l: label_map[int(l)]

#     y = np.asarray([normalize(l) for l in labels_in], dtype=np.int64)
#     return X, y

# def read_uea_ts(filename):
#     """
#     Robust reader for UEA .ts (multivariate or univariate).
#     - Ignores header lines starting with '@' (any case).
#     - Each data line: dim1 : dim2 : ... : label
#       where each dim is comma-separated floats (variable length allowed).
#     Returns:
#       X: list of np.ndarray, each shape (V, T_i)
#       y: list of raw labels (str)
#     """
#     X, y = [], []
#     with open(filename, "r") as f:
#         for raw in f:
#             line = raw.strip()
#             if not line or line.startswith("@") or line.startswith("#"):
#                 continue

#             parts = [p.strip() for p in line.split(":")]
#             if len(parts) < 2:
#                 continue

#             *dim_strs, label_str = parts
#             dims = []
#             for ds in dim_strs:
#                 if ds == "":
#                     continue
#                 toks = [t for t in ds.split(",") if t != ""]
#                 # coerce safely to float32
#                 row = []
#                 for t in toks:
#                     try:
#                         row.append(float(t))
#                     except Exception:
#                         row.append(0.0)
#                 dims.append(np.asarray(row, dtype=np.float32))

#             if not dims:
#                 continue

#             # pad each dimension to the same length within this sample
#             T = max(len(d) for d in dims)
#             dims = [np.pad(d, (0, T - len(d)), mode="edge") for d in dims]
#             sample = np.stack(dims, axis=0)  # (V, T)

#             X.append(sample)
#             y.append(label_str)
#     return X, y

# # helper for label mapping
# def _map_labels_to_int(train_labels, test_labels):
#     uniq = sorted(set(train_labels) | set(test_labels))
#     mapping = {lab: i for i, lab in enumerate(uniq)}
#     y_tr = np.array([mapping[l] for l in train_labels], dtype=np.int64)
#     y_te = np.array([mapping[l] for l in test_labels], dtype=np.int64)
#     return y_tr, y_te


# def stratified_val_test_split(dataset, labels, val_size=0.2, seed=42):
#     sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
#     indices = np.arange(len(dataset))
#     for val_idx, test_idx in sss.split(indices, labels):
#         return Subset(dataset, val_idx), Subset(dataset, test_idx)


# def build_chart_datasets(X, y, split, dataset_name, chart_branches, transform, generate_images=False, overwrite_existing=False, global_indices=None):
#     datasets = []
#     for branch_cfg in chart_branches.values():
#         ds = TimeSeriesImageDataset(
#             dataset_name=dataset_name,
#             time_series_data=X,
#             labels=y,
#             split=split,
#             chart_type=branch_cfg['chart_type'],
#             color_mode=branch_cfg.get('color_mode', 'color'),
#             label_mode=branch_cfg.get('label_mode', 'with_label'),
#             scatter_mode=branch_cfg.get('scatter_mode', 'plain'),
#             bar_mode=branch_cfg.get('bar_mode', 'border'),
#             transform=transform,
#             generate_images=generate_images,
#             overwrite_existing=overwrite_existing,
#             global_indices=global_indices if global_indices is not None else list(range(len(y))),
#             multivariate_mode=branch_cfg.get('multivariate_mode', 'subplots')  # NEW
#         )
#         datasets.append(ds)
#     return datasets



# def create_dataloaders(config, seed=42):
#     from collections import Counter
#     import numpy as np
#     from sklearn.model_selection import StratifiedShuffleSplit
#     from torchvision import transforms
#     from torch.utils.data import DataLoader

#     model_type = config['model']['type']
#     chart_branches = config.get('chart_branches', {})
#     dataset_cfg = config['dataset']
#     dataset_name = dataset_cfg['name']
#     dataset_format = dataset_cfg.get('format', 'ucr')  # 'ucr' | 'uea'
#     batch_size = config['training']['batch_size']

#     # Use ImageNet normalization only when running pretrained ResNet-18
#     pretrained = (
#         config['model'].get('chart_model', '').lower() == 'resnet18'
#         and config['model'].get('pretrained', False)
#     )

#     # Image size: keep it modest (128x128) for speed; bump to 224 if you want max transfer from pretrained
#     base_transforms = [
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#     ]
#     if pretrained:
#         base_transforms.append(
#             transforms.Normalize(mean=(0.485, 0.456, 0.406),
#                                  std=(0.229, 0.224, 0.225))
#         )
#     transform_train = transforms.Compose(base_transforms)
#     transform_eval = transforms.Compose(base_transforms)

#     # ---------- Load raw data ----------
#     if dataset_format == 'uea':
#         # Robust UEA reader must exist in this module:
#         # X_*_list: list of (V, Ti) float32 arrays (variable length); y_*_list: raw labels (str)
#         X_tr_list, y_tr_list = read_uea_ts(dataset_cfg['train_path'])
#         X_te_list, y_te_list = read_uea_ts(dataset_cfg['test_path'])

#         # Map labels to ints 0..C-1
#         y_train, y_test = _map_labels_to_int(y_tr_list, y_te_list)

#         # Global Tmax across train+test to allow fixed-length flatten for the numerical branch
#         def _tmax(lst):
#             return max(x.shape[1] for x in lst) if lst else 0

#         Tmax = max(_tmax(X_tr_list), _tmax(X_te_list))
#         if Tmax == 0:
#             raise ValueError(
#                 f"No data found in {dataset_cfg['train_path']} or {dataset_cfg['test_path']}"
#             )

#         # Pad each sample to (V, Tmax); keep charts as object array; also create flattened copies
#         def _pad_to_Tmax(lst, Tmax):
#             out = []
#             for x in lst:
#                 V, T = x.shape
#                 if T < Tmax:
#                     pad = np.tile(x[:, -1:], (1, Tmax - T))
#                     x = np.concatenate([x, pad], axis=1)
#                 out.append(x.astype(np.float32))
#             return np.array(out, dtype=object)

#         X_train = _pad_to_Tmax(X_tr_list, Tmax)  # object array of (V, Tmax)
#         X_test  = _pad_to_Tmax(X_te_list, Tmax)

#         # Flattened (V*Tmax,) float32 for numerical branch
#         X_train_flat = np.array([x.reshape(-1) for x in X_train], dtype=np.float32)
#         X_test_flat  = np.array([x.reshape(-1) for x in X_test],  dtype=np.float32)

#     else:
#         # UCR univariate; wrap each to (1, T) for a unified chart path
#         X_train_u, y_train = read_ucr(dataset_cfg['train_path'])
#         X_test_u,  y_test  = read_ucr(dataset_cfg['test_path'])

#         X_train = np.array([np.asarray(x, dtype=np.float32)[None, :] for x in X_train_u], dtype=object)
#         X_test  = np.array([np.asarray(x, dtype=np.float32)[None, :] for x in X_test_u],  dtype=object)

#         X_train_flat = np.array([x.reshape(-1) for x in X_train], dtype=np.float32)
#         X_test_flat  = np.array([x.reshape(-1) for x in X_test],  dtype=np.float32)

#     # Sanity
#     print("Train labels:", Counter(y_train))
#     print("Test labels:",  Counter(y_test))

#     # ---------- Build val/test split from the provided test set ----------
#     indices = np.arange(len(X_test))
#     if len(indices) == 0:
#         raise ValueError("Empty test split: cannot create val/test. Check dataset paths and parser.")

#     sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
#     val_idx, test_idx = next(sss.split(indices, y_test))
#     val_indices = indices[val_idx]
#     test_indices = indices[test_idx]

#     # Slice per split (note: X_* are object arrays of per-sample (V, T))
#     X_val,  y_val  = X_test[val_indices],  y_test[val_indices]
#     X_tst,  y_tst  = X_test[test_indices], y_test[test_indices]
#     X_val_flat     = X_test_flat[val_indices]
#     X_tst_flat     = X_test_flat[test_indices]

#     # ---------- Chart datasets ----------
#     chart_datasets = {
#         'train': build_chart_datasets(
#             X_train, y_train, 'train', dataset_name, chart_branches, transform_train,
#             generate_images=config['image_generation'].get('generate_images', False),
#             overwrite_existing=config['image_generation'].get('overwrite_existing', False)
#         ),
#         'val': build_chart_datasets(
#             X_val, y_val, 'test', dataset_name, chart_branches, transform_eval,
#             generate_images=config['image_generation'].get('generate_images', False),
#             overwrite_existing=config['image_generation'].get('overwrite_existing', False),
#             global_indices=val_indices
#         ),
#         'test': build_chart_datasets(
#             X_tst, y_tst, 'test', dataset_name, chart_branches, transform_eval,
#             generate_images=config['image_generation'].get('generate_images', False),
#             overwrite_existing=config['image_generation'].get('overwrite_existing', False),
#             global_indices=test_indices
#         ),
#     }

#     # ---------- Numerical datasets (for transformer/FCN/oscnn branch) ----------
#     numerical_datasets = {
#         'train': NumericalDataset(X_train_flat, y_train),
#         'val':   NumericalDataset(X_val_flat,   y_val),
#         'test':  NumericalDataset(X_tst_flat,   y_tst),
#     }

#     # ---------- Final DataLoaders ----------
#     dataloaders = {}
#     for split in ['train', 'val', 'test']:
#         shuffle = (split == 'train')

#         # One DataLoader per chart branch; for single-modal, we’ll collapse to a single loader below
#         chart_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
#                          for ds in chart_datasets[split]]

#         # Numerical only if model requests it
#         numerical_loader = None
#         if model_type in ['two_branch', 'multi_modal_chart'] and config['model'].get('numerical_branch', 'none') != 'none':
#             numerical_loader = DataLoader(numerical_datasets[split], batch_size=batch_size, shuffle=shuffle)

#         # For single_modal_chart: use the single chart loader directly (not a list)
#         if model_type == 'single_modal_chart':
#             chart_loaders = chart_loaders[0]

#         dataloaders[split] = {
#             'chart': chart_loaders,
#             'numerical': numerical_loader,
#         }

#     return dataloaders


import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter

from vtbench.data.chart_generator import TimeSeriesImageDataset, NumericalDataset


def read_ucr(filename):
    """
    Robust UCR/UEA-ish reader.
    Handles:
      - headers starting with '@' or '#'
      - label at END or START, numeric or string (and 'label:...' anywhere)
      - comma- or whitespace-delimited rows (mixed tolerated)
      - tokens like 'feat:1.23' -> 1.23
      - ragged series: auto-resamples each to median length
    Returns:
      X: np.ndarray of shape (N, T) float32
      y: np.ndarray of shape (N,) int64 in 0..C-1
    """
    import numpy as np

    def _is_int_like(s):
        try:
            f = float(s)
            return np.isfinite(f) and abs(f - round(f)) < 1e-6
        except Exception:
            return False

    def _is_float_like(s):
        try:
            f = float(s)
            return np.isfinite(f)
        except Exception:
            return False

    def _clean_tok(tok: str) -> str:
        # strip quotes and split on colon, keep rightmost
        t = tok.strip().strip("'").strip('"')
        if ":" in t:
            t = t.split(":")[-1].strip()
        return t

    def _split_line(line: str):
        # decide delimiter per-line; tolerate tabs
        if "," in line:
            parts = [p for p in line.split(",")]
        else:
            parts = line.replace("\t", " ").split()
        # trim empties
        return [p.strip() for p in parts if p.strip() != ""]

    def _resample_to_length(x: np.ndarray, target_len: int) -> np.ndarray:
        n = len(x)
        if n == target_len:
            return x.astype(np.float32, copy=False)
        if n <= 1:
            val = float(x[0]) if n == 1 else 0.0
            return np.full((target_len,), val, dtype=np.float32)
        old_idx = np.linspace(0.0, 1.0, num=n, dtype=np.float64)
        new_idx = np.linspace(0.0, 1.0, num=target_len, dtype=np.float64)
        y = np.interp(new_idx, old_idx, x.astype(np.float64, copy=False))
        return y.astype(np.float32, copy=False)

    data = []
    raw_labels = []

    with open(filename, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("@") or line.startswith("#"):
                continue

            line = line.rstrip(",")
            parts = _split_line(line)
            if len(parts) < 2:
                continue

            label_tok = None
            feat_parts = None

            # Try label at END
            last_tok = _clean_tok(parts[-1])
            if _is_float_like(last_tok) is False or last_tok.lower().isalpha():
                # last token is clearly non-numeric label
                label_tok = last_tok
                feat_parts = parts[:-1]
            elif _is_int_like(last_tok):
                # numeric label at END is common in some UCR CSVs
                label_tok = last_tok
                feat_parts = parts[:-1]

            if label_tok is None:
                # Try label at START
                first_tok = _clean_tok(parts[0])
                if (_is_float_like(first_tok) is False) or first_tok.lower().isalpha() or _is_int_like(first_tok):
                    # treat as label (string or numeric)
                    label_tok = first_tok
                    feat_parts = parts[1:]

            if label_tok is None:
                # Fallback: look for a token like 'label:XYZ'
                found_idx = -1
                for i, tok in enumerate(parts[::-1]):
                    if "label" in tok.lower():
                        cand = _clean_tok(tok)
                        if cand != "":
                            label_tok = cand
                            # remove exactly that token (use equality, not identity)
                            found_idx = len(parts) - 1 - i
                            break
                if label_tok is not None:
                    feat_parts = [t for j, t in enumerate(parts) if j != found_idx]

            if label_tok is None:
                # Give up on this row
                continue

            # Convert features safely
            feats = []
            for s in feat_parts:
                s_clean = _clean_tok(s)
                try:
                    v = float(s_clean)
                    if not np.isfinite(v):
                        v = 0.0
                    feats.append(v)
                except Exception:
                    # non-numerics in features -> drop row
                    feats = []
                    break

            if len(feats) == 0:
                continue

            data.append(feats)
            raw_labels.append(label_tok)

    if len(data) == 0:
        raise ValueError(f"No valid rows parsed from: {filename}")

    # Normalize labels (string or numeric) -> integers 0..C-1
    # First, canonicalize
    def _canon(l):
        ls = str(l).strip()
        ls = _clean_tok(ls)
        return ls

    labels_canon = [_canon(l) for l in raw_labels]

    # If all int-like, map by sorted ints; else map by sorted strings
    if all(_is_int_like(l) for l in labels_canon):
        nums = [int(round(float(l))) for l in labels_canon]
        uniq = sorted(set(nums))
        remap = {u: i for i, u in enumerate(uniq)}
        y = np.asarray([remap[n] for n in nums], dtype=np.int64)
    else:
        uniq = sorted(set(labels_canon))
        remap = {u: i for i, u in enumerate(uniq)}
        y = np.asarray([remap[l] for l in labels_canon], dtype=np.int64)

    # Handle ragged series by resampling to median length
    lengths = [len(x) for x in data]
    if len(set(lengths)) == 1:
        X = np.asarray(data, dtype=np.float32)
    else:
        target_len = int(np.median(lengths))
        if target_len < 1:
            target_len = max(1, max(lengths))
        X = np.stack(
            [_resample_to_length(np.asarray(x, dtype=np.float32), target_len) for x in data],
            axis=0
        )

    # Sanity: warn if collapsed class set
    if len(np.unique(y)) < 2:
        print(f"[WARN] Only one class detected in {filename}. "
              f"Parsed N={len(y)} rows, unique labels={sorted(set(labels_canon))[:10]}{'...' if len(set(labels_canon))>10 else ''}")

    return X, y

def read_uea_ts(filename):
    """
    Robust reader for UEA .ts (multivariate or univariate).
    - Ignores header lines starting with '@' or '#'.
    - Each data line: dim1 : dim2 : ... : label
      where each dim is comma-separated floats (variable length allowed).
    Returns:
      X: list of np.ndarray, each shape (V, T_i)
      y: list of raw labels (str)
    """
    X, y = [], []
    with open(filename, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("@") or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split(":")]
            if len(parts) < 2:
                continue

            *dim_strs, label_str = parts
            dims = []
            for ds in dim_strs:
                if ds == "":
                    continue
                toks = [t for t in ds.split(",") if t != ""]
                # coerce safely to float32
                row = []
                for t in toks:
                    try:
                        row.append(float(t))
                    except Exception:
                        row.append(0.0)
                dims.append(np.asarray(row, dtype=np.float32))

            if not dims:
                continue

            # pad each dimension to the same length within this sample
            T = max(len(d) for d in dims)
            dims = [np.pad(d, (0, T - len(d)), mode="edge") for d in dims]
            sample = np.stack(dims, axis=0)  # (V, T)

            X.append(sample)
            y.append(label_str)
    return X, y

def _label_sort_key(x):
    # numeric-first sort if possible, else lexicographic
    try:
        return (0, float(x))
    except Exception:
        return (1, str(x))


def _map_labels_to_int(train_labels, test_labels):
    """Map union of raw labels (strings) to 0..C-1; apply consistently to train/test."""
    tr = list(train_labels)
    te = list(test_labels)

    uniq = sorted(set(tr) | set(te), key=_label_sort_key)
    label_map = {lab: i for i, lab in enumerate(uniq)}

    y_train = np.asarray([label_map[l] for l in tr], dtype=np.int64)
    y_test  = np.asarray([label_map[l] for l in te], dtype=np.int64)
    return y_train, y_test, label_map


def stratified_val_test_split(dataset, labels, val_size=0.2, seed=42):
    """(Unused in the final pipeline) Kept for completeness."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    indices = np.arange(len(dataset))
    for val_idx, test_idx in sss.split(indices, labels):
        return Subset(dataset, val_idx), Subset(dataset, test_idx)


def build_chart_datasets(
    X,
    y,
    split,
    dataset_name,
    chart_branches,
    transform,
    generate_images=False,
    overwrite_existing=False,
    global_indices=None
):
    """
    Creates one TimeSeriesImageDataset per chart branch, sharing the same indices.
    """
    datasets = []
    for branch_cfg in chart_branches.values():
        ds = TimeSeriesImageDataset(
            dataset_name=dataset_name,
            time_series_data=X,
            labels=y,
            split=split,
            chart_type=branch_cfg['chart_type'],
            color_mode=branch_cfg.get('color_mode', 'color'),
            label_mode=branch_cfg.get('label_mode', 'with_label'),
            scatter_mode=branch_cfg.get('scatter_mode', 'plain'),
            bar_mode=branch_cfg.get('bar_mode', 'border'),
            transform=transform,
            generate_images=generate_images,
            overwrite_existing=overwrite_existing,
            global_indices=global_indices if global_indices is not None else list(range(len(y))),
            multivariate_mode=branch_cfg.get('multivariate_mode', 'subplots')
        )
        datasets.append(ds)
    return datasets


def create_dataloaders(config, seed=42):
    """
    Build dataloaders with:
      - Stratified validation split from TRAIN
      - TEST set kept intact
      - For UEA + preserve_multivariate_size: NO Resize (use native MV grid size)
      - Otherwise: Resize to 128x128
    """
    import numpy as np
    from collections import Counter
    from sklearn.model_selection import StratifiedShuffleSplit
    from torchvision import transforms
    from torch.utils.data import DataLoader

    from vtbench.data.chart_generator import TimeSeriesImageDataset, NumericalDataset
    # assumes you have read_ucr, read_uea_ts, _map_labels_to_int, build_chart_datasets available

    model_type    = config['model']['type']
    chart_branches = config.get('chart_branches', {})
    dataset_cfg   = config['dataset']
    dataset_name  = dataset_cfg['name']
    dataset_format = dataset_cfg.get('format', 'ucr')  # 'ucr' | 'uea'
    batch_size    = config['training']['batch_size']

    # Model flags
    chart_model = str(config['model'].get('chart_model', '')).lower()
    is_resnet18 = (chart_model == 'resnet18')
    pretrained  = (is_resnet18 and config['model'].get('pretrained', False))

    # Preserve native MV grid size for UEA?
    preserve_mv = bool(config.get('image_generation', {}).get('preserve_multivariate_size', True))

    # ---------- Transforms ----------
    # Univariate charts are saved at dynamic matplotlib size now.
    # -> Resize to 128x128 here (model input size), EXCEPT when we explicitly
    #    preserve multivariate (UEA) native grid pixels.
    if dataset_format == 'uea' and preserve_mv:
        base_transforms = [transforms.ToTensor()]   # keep exact MV pixels
    else:
        base_transforms = [
            transforms.Resize((128, 128)),          # unify size for models
            transforms.ToTensor()
        ]

    if pretrained:
        base_transforms.append(
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225))
        )

    transform_train = transforms.Compose(base_transforms)
    transform_eval  = transforms.Compose(base_transforms)

    # ---------- Load raw data ----------
    if dataset_format == 'uea':
        # UEA multivariate/univariate .ts → lists of (V, Ti) arrays + raw labels
        X_tr_list, y_tr_list = read_uea_ts(dataset_cfg['train_path'])
        X_te_list, y_te_list = read_uea_ts(dataset_cfg['test_path'])

        # Map union of labels → 0..C-1
        y_train, y_test, label_map = _map_labels_to_int(y_tr_list, y_te_list)

        # Pad to global Tmax (so numerical branch has fixed length V*Tmax)
        def _tmax(lst):
            return max(x.shape[1] for x in lst) if lst else 0

        Tmax = max(_tmax(X_tr_list), _tmax(X_te_list))
        if Tmax == 0:
            raise ValueError(f"No data found in {dataset_cfg['train_path']} or {dataset_cfg['test_path']}")

        def _pad_to_Tmax(lst, Tmax):
            out = []
            for x in lst:
                V, T = x.shape
                if T < Tmax:
                    pad = np.tile(x[:, -1:], (1, Tmax - T))
                    x = np.concatenate([x, pad], axis=1)
                out.append(x.astype(np.float32))
            return np.array(out, dtype=object)

        X_train = _pad_to_Tmax(X_tr_list, Tmax)  # object array of (V, Tmax)
        X_test  = _pad_to_Tmax(X_te_list, Tmax)

        # Flattened (V*Tmax,) float32 for numerical branch
        X_train_flat = np.array([x.reshape(-1) for x in X_train], dtype=np.float32)
        X_test_flat  = np.array([x.reshape(-1) for x in X_test],  dtype=np.float32)

    else:
        # UCR univariate; wrap each to (1, T) for unified chart path
        X_train_u, y_train = read_ucr(dataset_cfg['train_path'])
        X_test_u,  y_test  = read_ucr(dataset_cfg['test_path'])

        X_train = np.array([np.asarray(x, dtype=np.float32)[None, :] for x in X_train_u],
                           dtype=object)
        X_test  = np.array([np.asarray(x, dtype=np.float32)[None, :] for x in X_test_u],
                           dtype=object)

        X_train_flat = np.array([x.reshape(-1) for x in X_train], dtype=np.float32)
        X_test_flat  = np.array([x.reshape(-1) for x in X_test],  dtype=np.float32)

    # Sanity print
    print("Train labels:", Counter(y_train))
    print("Test  labels:", Counter(y_test))

    # ---------- Stratified val split from TRAIN; keep TEST intact ----------
    train_indices_all = np.arange(len(X_train))
    if len(train_indices_all) == 0:
        raise ValueError("Empty training split: cannot create train/val. Check dataset paths and parser.")

    val_size = config.get('training', {}).get('val_size', 0.2)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr_idx, val_idx = next(sss.split(train_indices_all, y_train))
    train_indices = train_indices_all[tr_idx]
    val_indices   = train_indices_all[val_idx]

    # TRAIN → (train, val); TEST untouched
    X_trn,  y_trn  = X_train[train_indices], y_train[train_indices]
    X_val,  y_val  = X_train[val_indices],   y_train[val_indices]
    X_trn_flat     = X_train_flat[train_indices]
    X_val_flat     = X_train_flat[val_indices]

    X_tst,  y_tst  = X_test,  y_test
    X_tst_flat     = X_test_flat

    # ---------- Chart datasets ----------
    chart_datasets = {
        'train': build_chart_datasets(
            X_trn, y_trn, 'train', dataset_name, chart_branches, transform_train,
            generate_images=config['image_generation'].get('generate_images', False),
            overwrite_existing=config['image_generation'].get('overwrite_existing', False),
            global_indices=train_indices.tolist()
        ),
        'val': build_chart_datasets(
            X_val, y_val, 'train', dataset_name, chart_branches, transform_eval,
            generate_images=config['image_generation'].get('generate_images', False),
            overwrite_existing=config['image_generation'].get('overwrite_existing', False),
            global_indices=val_indices.tolist()
        ),
        'test': build_chart_datasets(
            X_tst, y_tst, 'test', dataset_name, chart_branches, transform_eval,
            generate_images=config['image_generation'].get('generate_images', False),
            overwrite_existing=config['image_generation'].get('overwrite_existing', False),
            global_indices=list(range(len(y_tst)))
        ),
    }

    # ---------- Numerical datasets ----------
    numerical_datasets = {
        'train': NumericalDataset(X_trn_flat, y_trn),
        'val':   NumericalDataset(X_val_flat, y_val),
        'test':  NumericalDataset(X_tst_flat, y_tst),
    }

    # ---------- Final DataLoaders ----------
    dataloaders = {}
    for split in ['train', 'val', 'test']:
        shuffle = (split == 'train')

        # One DataLoader per chart branch; for single-modal, collapse to a single loader
        chart_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
                         for ds in chart_datasets[split]]

        numerical_loader = None
        if model_type in ['two_branch', 'multi_modal_chart'] and \
           config['model'].get('numerical_branch', 'none') != 'none':
            numerical_loader = DataLoader(
                numerical_datasets[split],
                batch_size=batch_size,
                shuffle=shuffle
            )

        if model_type == 'single_modal_chart':
            chart_loaders = chart_loaders[0]

        dataloaders[split] = {
            'chart': chart_loaders,
            'numerical': numerical_loader,
        }

    return dataloaders
