"""
Supp 6A: 30 KDD31 datasets × 17 encodings × DeepCNN × 3 seeds.

Encodings handled:
  - chart  : line, area, bar, scatter      → matplotlib via create_*_chart
  - math   : gasf, gadf, mtf, mtf_16, rp,
             rp_grayscale, cwt, stft         → ENCODING_REGISTRY
  - rgb    : gasf_gadf_mtf, rp_cwt_stft     → get_rgb_stack
  - colormap: gasf_viridis, gadf_plasma,
              rp_grayscale_inferno          → apply_colormap on grayscale base

Output schema mirrors original 6A:
  seed, dataset, encoding_type, encoding_name, model, accuracy,
  train_samples, test_samples, num_classes
"""
from __future__ import annotations

import argparse
import csv
import gc
import io
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.loader import read_ucr                                   # noqa: E402
from ts_image_encodings import (                              # noqa: E402
    get_encoding, get_rgb_stack, apply_colormap, RGB_STACK_PRESETS,
)
from data.chart_generator import (                                 # noqa: E402
    create_line_chart, create_area_chart, create_bar_chart,
    create_scatter_chart, TimeSeriesImageDataset,
)
from train.factory import get_chart_model                          # noqa: E402

CHART_IMAGE_ROOT = os.environ.get("CHART_IMAGE_ROOT", "/tmp/chart_images")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path("results/supp/s_ext_6a")
RESUME_CSV = OUT_DIR / "encoding_comparison.csv"
IMG_SIZE = 128
CHART_TYPES = {"line", "area", "bar", "scatter"}
COLORMAPS = ("viridis", "plasma", "inferno", "magma", "jet")

_CHART_FN = {
    "line": create_line_chart,
    "area": create_area_chart,
    "bar": create_bar_chart,
    "scatter": create_scatter_chart,
}


def _classify_encoding(name: str) -> str:
    if name in CHART_TYPES:
        return "chart"
    if name.startswith("rgb_") or name in RGB_STACK_PRESETS:
        return "rgb"
    if any(name.endswith(f"_{cm}") for cm in COLORMAPS):
        return "colormap"
    return "math"


def _render_chart(ts: np.ndarray, chart_type: str, image_size: int) -> np.ndarray:
    """Render chart image to a temp PNG, load as ndarray, return (3, H, W) float32."""
    import tempfile

    fn = _CHART_FN[chart_type]
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        path = tf.name
    try:
        # create_*_chart signatures: (ts, chart_path, [bar_mode|scatter_mode,]
        # color_mode, label_mode, global_y_range=None, *, linewidth, dpi)
        if chart_type == "bar":
            fn(ts, path, "border", "color", "with_label")
        elif chart_type == "scatter":
            fn(ts, path, "plain", "color", "with_label")
        else:
            fn(ts, path, "color", "with_label")
        img = Image.open(path).convert("RGB").resize((image_size, image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return np.transpose(arr, (2, 0, 1))
    finally:
        try: os.unlink(path)
        except Exception: pass
        # Free matplotlib figure backlog
        import matplotlib.pyplot as _plt
        _plt.close("all")


def _render_image(encoding: str, ts: np.ndarray, image_size: int) -> np.ndarray:
    """Return (3, H, W) float32 in [0, 1] for any supported encoding."""
    et = _classify_encoding(encoding)
    if et == "chart":
        return _render_chart(ts, encoding, image_size)
    if et == "rgb":
        preset = encoding.replace("rgb_", "")
        rgb = get_rgb_stack(preset, ts, image_size)
        arr = rgb.astype(np.float32) / 255.0
        return np.transpose(arr, (2, 0, 1))
    if et == "colormap":
        base, cmap = encoding.rsplit("_", 1)
        gray = get_encoding(base, ts, image_size)
        arr = apply_colormap(gray, cmap)
        if arr.ndim == 3:
            arr = arr.astype(np.float32) / 255.0
            return np.transpose(arr, (2, 0, 1))
        rgb = np.repeat(arr[None, :, :], 3, axis=0)
        return rgb.astype(np.float32) / 255.0
    # math
    img = get_encoding(encoding, ts, image_size)
    if img.ndim == 3:
        arr = img.astype(np.float32) / 255.0
        return np.transpose(arr, (2, 0, 1))
    rgb = np.repeat(img[None, :, :], 3, axis=0)
    return rgb.astype(np.float32) / 255.0


class _OnTheFlyDataset(Dataset):
    """For math/rgb/colormap encodings: generate via numpy at __getitem__."""
    def __init__(self, X, y, encoding, image_size=IMG_SIZE):
        self.X = X
        self.y = y
        self.encoding = encoding
        self.image_size = image_size

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        ts = np.asarray(self.X[idx], dtype=np.float64).ravel()
        arr = _render_image(self.encoding, ts, self.image_size)
        return torch.from_numpy(arr).float(), int(self.y[idx])


class _CachedChartDataset(Dataset):
    """For chart encodings: load pre-generated PNGs from /tmp/chart_images.

    Filenames follow vtbench.data.chart_generator naming convention:
      {CHART_DIR_MAP[chart_type]}_with_label/{split}/{prefix}_{global_idx}.png
    """
    _CHART_DIR_MAP = {
        "line": "line_charts_color_with_label",
        "area": "area_charts_color_with_label",
        "bar": "bar_charts_border_color_with_label",
        "scatter": "scatter_charts_plain_color_with_label",
    }
    _PREFIX = {
        "line": "line_chart_color_with_label",
        "area": "area_chart_color_with_label",
        "bar": "bar_chart_border_color_with_label",
        "scatter": "scatter_chart_plain_color_with_label",
    }

    def __init__(self, dataset_name, split, chart_type, global_indices,
                 labels, image_size=IMG_SIZE):
        self.global_indices = global_indices
        self.labels = labels
        self.image_size = image_size
        self.img_dir = os.path.join(
            CHART_IMAGE_ROOT,
            f"{dataset_name}_images",
            self._CHART_DIR_MAP[chart_type],
            split,
        )
        self.prefix = self._PREFIX[chart_type]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        gidx = self.global_indices[idx]
        path = os.path.join(self.img_dir, f"{self.prefix}_{gidx}.png")
        img = Image.open(path).convert("RGB").resize((self.image_size, self.image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(np.transpose(arr, (2, 0, 1))).float(), int(self.labels[idx])


def _ensure_chart_images(dataset_name, chart_type,
                         X_train, y_train, X_test, y_test,
                         train_indices, test_indices):
    """Pre-generate all chart PNGs for one (dataset, chart_type) if missing.
    Idempotent: TimeSeriesImageDataset skips existing files.
    """
    chart_dir = _CachedChartDataset._CHART_DIR_MAP[chart_type]
    base = os.path.join(CHART_IMAGE_ROOT, f"{dataset_name}_images")
    train_dir = os.path.join(base, chart_dir, "train")
    test_dir = os.path.join(base, chart_dir, "test")
    prefix = _CachedChartDataset._PREFIX[chart_type]

    def _all_present(d, indices):
        if not os.path.isdir(d):
            return False
        for i in indices:
            if not os.path.exists(os.path.join(d, f"{prefix}_{i}.png")):
                return False
        return True

    if _all_present(train_dir, train_indices) and _all_present(test_dir, test_indices):
        return

    print(f"  Generating {chart_type} for {dataset_name}...", flush=True)
    for split, X_s, y_s, g_idx in [
        ("train", X_train, y_train, train_indices),
        ("test", X_test, y_test, test_indices),
    ]:
        TimeSeriesImageDataset(
            time_series_data=list(X_s),
            labels=list(y_s),
            dataset_name=dataset_name,
            split=split,
            chart_type=chart_type,
            color_mode="color",
            label_mode="with_label",
            generate_images=True,
            global_indices=g_idx,
        )


_FIELDS = ["seed", "dataset", "encoding_type", "encoding_name", "model",
           "accuracy", "train_samples", "test_samples", "num_classes",
           "train_time_s", "flag_crashed", "note"]


def _load_completed_keys():
    if not RESUME_CSV.exists():
        return set()
    with open(RESUME_CSV) as f:
        rows = list(csv.DictReader(f))
    return {(r["dataset"], r["encoding_name"], str(r["seed"]))
            for r in rows if r.get("flag_crashed") == "0"}


def _append_row(row):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not RESUME_CSV.exists()
    with open(RESUME_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)


def _run_one(cfg, dataset, encoding, seed,
             X_train=None, y_train=None, X_test=None, y_test=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if X_train is None:
        train_path = Path("UCRArchive_2018") / dataset / f"{dataset}_TRAIN.tsv"
        test_path = Path("UCRArchive_2018") / dataset / f"{dataset}_TEST.tsv"
        X_train, y_train = read_ucr(str(train_path))
        X_test, y_test = read_ucr(str(test_path))
    num_classes = len(set(y_train.tolist()) | set(y_test.tolist()))

    enc_type = _classify_encoding(encoding)
    if enc_type == "chart":
        n_train = len(X_train); n_test = len(X_test)
        train_indices = list(range(n_train))
        test_indices = list(range(n_train, n_train + n_test))
        _ensure_chart_images(dataset, encoding, X_train, y_train,
                             X_test, y_test, train_indices, test_indices)
        train_ds = _CachedChartDataset(dataset, "train", encoding,
                                       train_indices, list(y_train),
                                       cfg["image_size"])
        test_ds = _CachedChartDataset(dataset, "test", encoding,
                                      test_indices, list(y_test),
                                      cfg["image_size"])
    else:
        train_ds = _OnTheFlyDataset(X_train, y_train, encoding, cfg["image_size"])
        test_ds = _OnTheFlyDataset(X_test, y_test, encoding, cfg["image_size"])

    bs = min(cfg["batch_size"], len(train_ds)) if len(train_ds) > 0 else 1
    # Adaptive bs to ensure ≥2 batches per epoch on small datasets
    safe_bs = max(1, len(train_ds) // 2) if bs > len(train_ds) // 2 else bs
    train_loader = DataLoader(
        train_ds, batch_size=safe_bs, shuffle=True, drop_last=True,
        num_workers=0, pin_memory=False)
    test_loader = DataLoader(
        test_ds, batch_size=min(cfg["batch_size"], len(test_ds)),
        shuffle=False, num_workers=0, pin_memory=False)

    if len(train_loader) == 0:
        return dict(accuracy=-1.0, train_time_s=0.0, flag_crashed=1,
                    note="empty_train_loader")

    model = get_chart_model(
        "deepcnn", input_channels=3, num_classes=num_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                           weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    t0 = time.time()
    best_acc = 0.0
    patience_counter = 0
    for epoch in range(cfg["epochs"]):
        model.train()
        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            out = model(imgs)
            loss = criterion(out, labels)
            if not torch.isfinite(loss):
                return dict(accuracy=-1.0, train_time_s=time.time()-t0,
                            flag_crashed=1, note="loss_nan")
            loss.backward()
            opt.step()
        # Quick val on test
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                out = model(imgs)
                correct += (out.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
        acc = correct / total if total else 0.0
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break
    train_time = time.time() - t0
    return dict(accuracy=best_acc, train_time_s=round(train_time, 2),
                flag_crashed=0, note="")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg_yaml = yaml.safe_load(f)
    exp = cfg_yaml["experiment"]
    tr = cfg_yaml["training"]

    datasets = exp["datasets"]
    encodings = (exp.get("chart_encodings", []) +
                 exp.get("math_encodings", []) +
                 [f"rgb_{x}" for x in exp.get("rgb_encodings", [])] +
                 exp.get("colormap_encodings", []))
    seeds = exp["seeds"]

    run_cfg = dict(
        image_size=exp.get("image_size", IMG_SIZE),
        epochs=tr["epochs"],
        learning_rate=tr["learning_rate"],
        batch_size=tr["batch_size"],
    )

    completed = _load_completed_keys()
    total = len(datasets) * len(encodings) * len(seeds)
    idx = 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for ds in datasets:
        # Load once per dataset to cache num_classes / train_samples
        train_path = Path("UCRArchive_2018") / ds / f"{ds}_TRAIN.tsv"
        test_path = Path("UCRArchive_2018") / ds / f"{ds}_TEST.tsv"
        X_train, y_train = read_ucr(str(train_path))
        X_test, y_test = read_ucr(str(test_path))
        n_train = len(X_train); n_test = len(X_test)
        num_classes = len(set(y_train.tolist()) | set(y_test.tolist()))

        for enc in encodings:
            enc_type = _classify_encoding(enc)
            for seed in seeds:
                idx += 1
                key = (ds, enc, str(seed))
                tag = f"[{idx}/{total}] {ds} | {enc_type}/{enc} | seed={seed}"
                if key in completed:
                    print(f"{tag}  SKIP (done)", flush=True)
                    continue
                print(tag, flush=True)
                try:
                    r = _run_one(run_cfg, ds, enc, seed,
                                 X_train=X_train, y_train=y_train,
                                 X_test=X_test, y_test=y_test)
                    print(f"  -> acc={r['accuracy']:.4f} time={r['train_time_s']:.1f}s",
                          flush=True)
                except Exception as e:
                    tb = traceback.format_exc()
                    print(f"  -> CRASHED: {type(e).__name__}: {e}", flush=True)
                    print(tb[-500:], flush=True)
                    r = dict(accuracy=-1.0, train_time_s=0.0,
                             flag_crashed=1,
                             note=f"{type(e).__name__}:{str(e)[:80]}")
                _append_row({
                    "seed": seed,
                    "dataset": ds,
                    "encoding_type": enc_type,
                    "encoding_name": enc,
                    "model": "deepcnn",
                    "accuracy": r["accuracy"],
                    "train_samples": n_train,
                    "test_samples": n_test,
                    "num_classes": num_classes,
                    "train_time_s": r["train_time_s"],
                    "flag_crashed": r["flag_crashed"],
                    "note": r["note"],
                })
                gc.collect()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

    print(f"\nDone. CSV at {RESUME_CSV}", flush=True)


if __name__ == "__main__":
    main()
