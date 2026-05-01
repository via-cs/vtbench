"""
Evaluates and compares test accuracy of:
  - baseline model (trained_model.pth)
  - HINT fine-tuned model (hint_finetuned_model.pth)

for each dataset / encoding combination.

Output: a CSV and a printed table with before/after accuracy and delta.

Usage:
    python evaluate_hint.py
"""

import os
import re
import csv
from typing import Dict, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms as T

from models.chart_models.deepcnn import DeepCNN


# Config
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE    = 124
GRAYSCALE_INPUT = False

DATA_ROOT    = "/Users/akkumy/Downloads/test_vtbench/vtbench/data"
RESULTS_ROOT = "/Users/akkumy/Downloads/test_vtbench/vtbench/vtbench"
IMAGE_ROOT   = "chart_images"

ENCODING_TYPES = ["line", "bar", "area", "scatter"]

chart_config: Dict[str, str] = {
    "area":    "_charts_color_with_label",
    "line":    "_charts_color_with_label",
    "bar":     "_charts_border_color_with_label",
    "scatter": "_charts_plain_color_with_label",
}

DATASET_NAMES = ["PhalangesOutlinesCorrect","ChlorineConcentration","SonyAIBORobotSurface1","Adiac","FaceAll","FacesUCR",
                "ArrowHead","CricketX","CricketY","CricketZ","InsectWingBeat","ToeSegmentation1","ToeSegmentation2",
                "Wine","WordSynonyms","Beef","BeetleFly","Computers","Earthquakes","Ham","Herring","RefrigerationDevices",
                "SharePriceIncrease","Crop","ECG5000","GunPoint","Lightning2","Strawberry","Yoga","FordB","Wafer"]

OUTPUT_CSV = "hint_accuracy_comparison.csv"



def sort_key_numeric(filename: str) -> int:
    """
    Extract trailing integer from VTBench chart filenames so images are
    sorted in .ts row order
    e.g. 'line_chart_color_with_label_42.png' -> 42
    """
    match = re.search(r'_(\d+)\.(png|jpg|jpeg)$', filename, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def load_ts_labels(ts_path: str) -> list:
    labels_raw = []
    in_data    = False
    for encoding in ("utf-8", "latin-1"):
        try:
            with open(ts_path, "r", encoding=encoding) as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.lower() == "@data":
                        in_data = True
                        continue
                    if in_data:
                        labels_raw.append(line.split(":")[-1].strip())
            break
        except UnicodeDecodeError:
            labels_raw = []
            in_data    = False
            continue

    if not labels_raw:
        raise RuntimeError(f"Could not read labels from {ts_path}")

    unique = sorted(set(labels_raw))
    try:
        unique = sorted(unique, key=lambda x: int(x))
    except ValueError:
        pass

    label_map = {s: i for i, s in enumerate(unique)}
    print(f"  Raw label sample: {labels_raw[:5]}")
    print(f"  Label map: {label_map}")  
    return [label_map[lbl] for lbl in labels_raw]



# Image loading
def load_image_tensor(image_path: str) -> torch.Tensor:
    pil_img = Image.open(image_path).convert("RGB")
    if GRAYSCALE_INPUT:
        pil_img = pil_img.convert("L").convert("RGB")
    pil_img = pil_img.resize((IMG_SIZE, IMG_SIZE))
    inp = T.ToTensor()(pil_img).unsqueeze(0).to(DEVICE)
    return inp


# Model loading
def get_last_conv_layer(m: nn.Module) -> nn.Module:
    convs = [mod for mod in m.modules() if isinstance(mod, nn.Conv2d)]
    if not convs:
        raise RuntimeError("No Conv2d layer found.")
    return convs[-1]


def load_model(model_path: str, num_classes: int) -> nn.Module:
    model = DeepCNN(input_channels=3, num_classes=num_classes).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


# Evaluation
def evaluate(model: nn.Module, image_paths: List[str], labels: List[int]) -> float:
    """Returns accuracy as a percentage"""
    model.eval()
    correct = 0
    with torch.no_grad():
        for img_path, lbl in zip(image_paths, labels):
            try:
                inp  = load_image_tensor(img_path)
                pred = model(inp).argmax(dim=1).item()
                correct += int(pred == lbl)
            except Exception as e:
                print(f"Skipping {img_path}: {e}")
    return 100.0 * correct / max(len(labels), 1)



def main():
    results = []

    for dataset_name in DATASET_NAMES:
        ts_path = os.path.join(DATA_ROOT, dataset_name, f"{dataset_name}_TEST.ts")
        if not os.path.exists(ts_path):
            print(f"[SKIP] {dataset_name} — TEST .ts not found: {ts_path}")
            continue

        try:
            labels = load_ts_labels(ts_path)
        except Exception as e:
            print(f"[SKIP] {dataset_name} — could not load labels: {e}")
            continue

        num_classes = len(set(labels))

        for encoding_type in ENCODING_TYPES:
            print(f"\n{dataset_name} / {encoding_type}")

            test_img_dir = os.path.join(
                IMAGE_ROOT,
                f"{dataset_name}_images",
                f"{encoding_type}{chart_config[encoding_type]}",
                "test",
            )

            if not os.path.exists(test_img_dir):
                print(f"[SKIP] test image dir not found: {test_img_dir}")
                continue

            image_paths = sorted(
                [
                    os.path.join(test_img_dir, f)
                    for f in os.listdir(test_img_dir)
                    if os.path.splitext(f)[1].lower() in {".png", ".jpg", ".jpeg"}
                ],
                key=lambda p: sort_key_numeric(os.path.basename(p))
            )

            if not image_paths:
                print(f"[SKIP] no images found in {test_img_dir}")
                continue

            if len(image_paths) != len(labels):
                print(f"[SKIP] image count ({len(image_paths)}) != label count ({len(labels)})")
                continue

            baseline_path   = os.path.join(RESULTS_ROOT, f"{dataset_name}_results",
                                            "single_chart", encoding_type, "trained_model.pth")
            hint_model_path = os.path.join(RESULTS_ROOT, f"{dataset_name}_results",
                                            "single_chart", encoding_type, "hint_finetuned_model.pth")

            if not os.path.exists(baseline_path):
                print(f"[SKIP] baseline model not found: {baseline_path}")
                continue

            if not os.path.exists(hint_model_path):
                print(f"[SKIP] HINT model not found: {hint_model_path}")
                continue

            try:
                baseline_model = load_model(baseline_path,   num_classes)
                hint_model     = load_model(hint_model_path, num_classes)
            except Exception as e:
                print(f"[SKIP] model load error: {e}")
                continue
            with torch.no_grad():
                for p, l in zip(image_paths[:3], labels[:3]):
                    inp  = load_image_tensor(p)
                    pred = baseline_model(inp).argmax(dim=1).item()
                    print(f"  {os.path.basename(p)} → label={l}, pred={pred}")

            baseline_acc = evaluate(baseline_model, image_paths, labels)
            hint_acc     = evaluate(hint_model,     image_paths, labels)
            delta        = hint_acc - baseline_acc

            print(f"Baseline : {baseline_acc:.2f}%")
            print(f"HINT     : {hint_acc:.2f}%")
            print(f"Delta    : {delta:+.2f}%")

            results.append({
                "dataset":      dataset_name,
                "encoding":     encoding_type,
                "baseline_acc": round(baseline_acc, 2),
                "hint_acc":     round(hint_acc, 2),
                "delta":        round(delta, 2),
            })

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "encoding",
                                                "baseline_acc", "hint_acc", "delta"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()