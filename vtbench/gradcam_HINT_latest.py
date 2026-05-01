"""
1) Loads VTBench chart model and generates Grad-CAM overlays

2) Adds "masked region" image generation:
   - occlude-only (mask only the important region) white boxing

3) HINT-style fine-tuning portion:
   - Grad-CAM serves as a gradient-based importance map as the model explanation.
   - Preserves task performance with the normal classification loss.
"""

import os
import re
import traceback
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

# model
from models.chart_models.deepcnn import DeepCNN

# Grad-CAM library (used for overlays / debugging)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# masking before proceeding to HINT
from masking_utils import cam_to_bbox, apply_box_mask


# ----------------------------
# Config
# ----------------------------

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE      = 128
GRAYSCALE_INPUT = False

ENCODING_TYPES = ["line", "bar", "area", "scatter"]

chart_config: Dict[str, str] = {
    "area":    "_charts_color_with_label",
    "line":    "_charts_color_with_label",
    "bar":     "_charts_border_color_with_label",
    "scatter": "_charts_plain_color_with_label",
}

# Root paths
DATA_ROOT    = "/Users/akkumy/Downloads/test_vtbench/vtbench/data"
RESULTS_ROOT = "/Users/akkumy/Downloads/test_vtbench/vtbench/vtbench"
IMAGE_ROOT   = "chart_images"
DATASET_NAMES = ["PhalangesOutlinesCorrect","ChlorineConcentration","SonyAIBORobotSurface1","Adiac","FaceAll","FacesUCR",
                "ArrowHead","CricketX","CricketY","CricketZ","InsectWingBeat","ToeSegmentation1","ToeSegmentation2",
                "Wine","WordSynonyms","Beef","BeetleFly","Computers","Earthquakes","Ham","Herring","RefrigerationDevices",
                "SharePriceIncrease","Crop","ECG5000","GunPoint","Lightning2","Strawberry","Yoga","FordB","Wafer"]



# MODEL_PATHS[dataset][encoding] -> path to trained_model.pth
MODEL_PATHS: Dict[str, Dict[str, str]] = {
    ds: {
        enc: os.path.join(RESULTS_ROOT, f"{ds}_results", "single_chart", enc, "trained_model.pth")
        for enc in ENCODING_TYPES
    }
    for ds in DATASET_NAMES
}

# TS_TRAIN_PATHS[dataset] -> path to <Dataset>_TRAIN.ts
TS_TRAIN_PATHS: Dict[str, str] = {
    ds: os.path.join(DATA_ROOT, ds, f"{ds}_TRAIN.ts")
    for ds in DATASET_NAMES
}

# We use WHITE box variant as the pseudo "human attention"
ATT_SUFFIX   = "_occlude_white.png"
WHITE_THRESH = 0.95  # pixels brighter than this count as "white box"

OUTPUT_DIR     = "gradcam_overlays_hint_integrated"
os.makedirs(OUTPUT_DIR, exist_ok=True)
ATTENTION_ROOT = OUTPUT_DIR


# ----------------------------
# Label loading
# ----------------------------

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
        raise RuntimeError(f"Could not read labels from {ts_path} — tried utf-8 and latin-1")

    unique = sorted(set(labels_raw))

    try:
        unique = sorted(unique, key=lambda x: int(x))
    except ValueError:
        pass

    label_map = {s: i for i, s in enumerate(unique)}
    return [label_map[lbl] for lbl in labels_raw]


# ----------------------------
# Model loading
# ----------------------------

def get_last_conv_layer(m: nn.Module) -> nn.Module:
    """Return last Conv2d layer (used as Grad-CAM target layer)."""
    convs = [mod for mod in m.modules() if isinstance(mod, nn.Conv2d)]
    if not convs:
        raise RuntimeError("No Conv2d layer found in the model.")
    return convs[-1]


def load_model_for_encoding(dataset_name: str, encoding_type: str,
                             num_classes: int) -> Tuple[nn.Module, nn.Module]:
    """
    Loads DeepCNN weights for a given dataset/encoding and returns
    (model, target_layer_for_cam). num_classes is inferred from the .ts file.
    """
    model_path = MODEL_PATHS.get(dataset_name, {}).get(encoding_type)
    if model_path is None:
        raise ValueError(f"No MODEL_PATH configured for {dataset_name}/{encoding_type}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = DeepCNN(input_channels=3, num_classes=num_classes).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    target_layer = get_last_conv_layer(model)
    return model, target_layer


# ----------------------------
# Image loading
# ----------------------------

def load_image_tensor(image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Loads image and returns:
      - input_tensor: [1,3,H,W] float in [0,1]
      - rgb_np: HxWx3 float in [0,1] for overlay visualization
    """
    pil_img = Image.open(image_path).convert("RGB")
    if GRAYSCALE_INPUT:
        pil_img = pil_img.convert("L").convert("RGB")
    pil_img_resized = pil_img.resize((IMG_SIZE, IMG_SIZE))
    rgb_np = np.array(pil_img_resized, dtype=np.float32) / 255.0

    transform = T.Compose([T.ToTensor()])
    inp = transform(pil_img_resized).unsqueeze(0).to(DEVICE)
    return inp, rgb_np


def attention_path_for_image(image_path: str, attention_root: str) -> str:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(attention_root, f"{stem}_mask.png")


'''def load_attention_mask(attention_path: str) -> torch.Tensor:
    """
    Loads a grayscale attention mask and returns tensor shape [1,1,H,W] in {0,1}.
    """
    att = cv2.imread(attention_path, cv2.IMREAD_GRAYSCALE)
    if att is None:
        raise FileNotFoundError(f"Attention map not found: {attention_path}")
    att = cv2.resize(att, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    att = att.astype(np.float32) / 255.0
    att = (att > 0.5).astype(np.float32)
    return torch.from_numpy(att)[None, None, :, :].to(DEVICE)'''


# ----------------------------
# Grad-CAM (overlay visualization portion)
# ----------------------------

def gradcam_map_different(
    model: nn.Module,
    target_layer: nn.Module,
    x: torch.Tensor,
    class_idx: Optional[int] = None,
) -> torch.Tensor:

    activations = None
    gradients   = None

    def fwd_hook(_, __, out):
        nonlocal activations
        activations = out

    def bwd_hook(_, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    logits = model(x)
    if class_idx is None:
        class_idx = int(torch.argmax(logits, dim=1).item())

    score = logits[:, class_idx].sum()
    model.zero_grad(set_to_none=True)
    torch.autograd.grad(score, activations, retain_graph=True, create_graph=True)

    assert activations is not None and gradients is not None, "Hooks did not capture tensors."

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam     = (weights * activations).sum(dim=1, keepdim=False)
    cam     = F.relu(cam)

    cam_min = cam.amin(dim=(1, 2), keepdim=True)
    cam_max = cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    cam     = (cam - cam_min) / (cam_max - cam_min)

    cam = cam.unsqueeze(1)
    cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    cam = cam.squeeze(1)

    h1.remove()
    h2.remove()
    return cam[0]  # [H,W] format


# ----------------------------
# Mask generation + boxing
# ----------------------------

def cam_to_bbox(cam: np.ndarray, topk_frac: float = 0.15) -> Tuple[int, int, int, int]:
    h, w  = cam.shape
    flat  = cam.reshape(-1)
    k     = max(1, int(topk_frac * flat.size))

    thresh = np.partition(flat, -k)[-k]
    mask   = (cam >= thresh).astype(np.uint8)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, 0, w, h

    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    return int(x1), int(y1), int(x2), int(y2)


def apply_box_mask(
    rgb_np: np.ndarray,
    bbox: Tuple[int, int, int, int],
    mode: str,
    fill_value: float,
) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    out = rgb_np.copy()
    if mode == "keep":
        out[:] = fill_value
        out[y1:y2, x1:x2, :] = rgb_np[y1:y2, x1:x2, :]
    elif mode == "occlude":
        out[y1:y2, x1:x2, :] = fill_value
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return out


# ----------------------------
# HINT losses
# ----------------------------

def ranking_loss_l1(cam: torch.Tensor, annotation_mask: torch.Tensor) -> torch.Tensor:
    """
    L1 p-norm distance between CAM and binary human mask.
    cam:        [1,1,H,W]
    annotation_mask: also [1,1,H,W]
    """
    hm = annotation_mask.squeeze(0).squeeze(0)
    c  = cam.squeeze(0).squeeze(0)
    return torch.mean(torch.abs(c - hm))


def pairwise_margin_loss(
    cam: torch.Tensor,
    annotation_mask: torch.Tensor,
    margin: float = 0.1,
    num_samples: int = 256,
) -> torch.Tensor:
    """
    Pairwise ranking loss: CAM(positive pixel) >= CAM(negative pixel) + margin.
    """
    hm= annotation_mask.squeeze(0).squeeze(0)
    cam_flat = cam.squeeze(0).squeeze(0).flatten()
    hm_flat  = hm.flatten()

    pos_idx = torch.nonzero(hm_flat > 0.5,  as_tuple=False).flatten()
    neg_idx = torch.nonzero(hm_flat <= 0.5, as_tuple=False).flatten()

    if pos_idx.numel() == 0 or neg_idx.numel() == 0:
        return cam_flat.new_tensor(0.0)

    ps = pos_idx[torch.randint(0, pos_idx.numel(), (min(num_samples, pos_idx.numel()),), device=cam.device)]
    ns = neg_idx[torch.randint(0, neg_idx.numel(), (min(num_samples, neg_idx.numel()),), device=cam.device)]

    pos_vals = cam_flat[ps]
    neg_vals = cam_flat[ns]

    diff = pos_vals[:, None] - neg_vals[None, :]
    return F.relu(margin - diff).mean()


# ----------------------------
# HINT training config + dataset
# ----------------------------

@dataclass
class HintTrainConfig:
    epochs:           int   = 100
    batch_size:       int   = 20
    lr:               float = 1e-4
    lambda_hint:      float = 1.0
    margin:           float = 0.1
    num_pair_samples: int   = 256


class ChartWithAttentionDataset(Dataset):

    def __init__(self, image_paths, labels, att_root, dataset_name, encoding_type):
        self.image_paths   = image_paths
        self.labels        = labels
        self.att_root      = att_root
        self.dataset_name  = dataset_name
        self.encoding_type = encoding_type

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        y = self.labels[idx]

        # original chart image
        pil = Image.open(p).convert("RGB")
        if GRAYSCALE_INPUT:
            pil = pil.convert("L").convert("RGB")
        pil = pil.resize((IMG_SIZE, IMG_SIZE))
        img = np.asarray(pil).astype(np.float32) / 255.0
        x   = torch.from_numpy(img).permute(2, 0, 1)

        # corresponding _occlude_white.png 
        stem       = os.path.splitext(os.path.basename(p))[0]
        boxed_name = f"{self.dataset_name}_{self.encoding_type}_{stem}{ATT_SUFFIX}"
        boxed_path = os.path.join(self.att_root, boxed_name)

        if not os.path.exists(boxed_path):
            raise FileNotFoundError(f"Missing white-box attention file:\n{boxed_path}")

        #conversion to binary mask
        boxed      = Image.open(boxed_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        boxed_np   = np.asarray(boxed).astype(np.float32) / 255.0
        whiteness  = boxed_np.mean(axis=2)     
        annotation_mask = (whiteness > WHITE_THRESH).astype(np.float32)
        annotation_mask = torch.from_numpy(annotation_mask).unsqueeze(0)    

        return x, torch.tensor(int(y), dtype=torch.long), annotation_mask


# ----------------------------
# Differentiable Grad-CAM (used inside HINT training loop)
# ----------------------------

def gradcam_map_differentiable(model, target_layer, x, class_idx=None):
    """
    Differentiable Grad-CAM for input "x"

    Args:
        model:        nn.Module
        target_layer: the conv layer to hook
        x:            torch.Tensor [B, 3, H, W]
        class_idx:    None(default), Tensor[B]

    Returns:
        cam: torch.Tensor [B, 1, H, W] normalised to [0,1]
    """
    activations = []
    gradients   = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def full_backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(full_backward_hook)

    logits = model(x)

    if class_idx is None:
        class_idx = logits.argmax(dim=1)
    elif isinstance(class_idx, int):
        class_idx = torch.tensor([class_idx] * logits.size(0), device=logits.device)
    elif isinstance(class_idx, (list, tuple)):
        class_idx = torch.tensor(class_idx, device=logits.device)
    else:
        class_idx = class_idx.to(logits.device)
        if class_idx.ndim == 0:
            class_idx = class_idx.view(1)
        if class_idx.numel() == 1 and logits.size(0) > 1:
            class_idx = class_idx.repeat(logits.size(0))
    class_idx = class_idx.long()

    selected = logits.gather(1, class_idx.unsqueeze(1)).squeeze(1)

    model.zero_grad(set_to_none=True)
    selected.sum().backward(retain_graph=True)

    A  = activations[0]
    dA = gradients[0]

    weights  = dA.mean(dim=(2, 3), keepdim=True)
    cam      = (weights * A).sum(dim=1, keepdim=True)
    cam      = torch.relu(cam)
    cam      = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)

    cam_flat = cam.view(cam.size(0), -1)
    cam_min  = cam_flat.min(dim=1)[0].view(-1, 1, 1, 1)
    cam_max  = cam_flat.max(dim=1)[0].view(-1, 1, 1, 1)
    cam      = (cam - cam_min) / (cam_max - cam_min + 1e-8)

    fh.remove()
    bh.remove()
    return cam 


def set_batchnorm_eval(m):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        m.eval()


# ----------------------------
# HINT fine-tuning loop
# ----------------------------

def hint_finetune(
    model: nn.Module,
    target_layer: nn.Module,
    loader: DataLoader,
    cfg: HintTrainConfig,
) -> None:
    """
    Fine-tune model with:
        L = CE(logits, y) + lambda_hint * (ranking_loss_l1 + pairwise_margin_loss)

    loader yields (x [B,3,H,W], y [B], and annotation_mask [B,1,H,W])
    """
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for ep in range(cfg.epochs):
        running = 0.0

        for x, y, annotation_masks in loader:
            x           = x.to(DEVICE)            
            y           = y.to(DEVICE)            
            annotation_masks = annotation_masks.to(DEVICE)  

            logits = model(x)
            ce     = F.cross_entropy(logits, y)

            hint_total = ce.new_tensor(0.0)
            B, C, H, W = x.shape

            for i in range(B):
                xi         = x[i:i+1]                  
                yi         = int(y[i].item())
                annotation_mask = annotation_masks[i].unsqueeze(0) 

                model.apply(set_batchnorm_eval)
                cam_orig = gradcam_map_differentiable(model, target_layer, xi, class_idx=yi)
                cam_np   = cam_orig[0, 0].detach().cpu().numpy()
                rgb_np   = xi[0].detach().permute(1, 2, 0).cpu().numpy()

                # Apply box mask
                bbox      = cam_to_bbox(cam_np, topk_frac=BOX_TOPK_FRAC)
                boxed_rgb = apply_box_mask(rgb_np, bbox=bbox, mode=BOX_MODE, fill_value=BOX_FILL_VALUE)
                boxed_x   = (
                    torch.from_numpy(boxed_rgb)
                    .permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
                )  

                # Grad-CAM on boxed image
                cam_boxed = gradcam_map_differentiable(model, target_layer, boxed_x, class_idx=yi)

                # HINT losses against pre-computed mask
                rloss = ranking_loss_l1(cam_boxed, annotation_mask)
                ploss = pairwise_margin_loss(
                    cam_boxed, annotation_mask,
                    margin=cfg.margin,
                    num_samples=cfg.num_pair_samples,
                )
                hint_total = hint_total + (rloss + ploss)

            hint_loss = hint_total / float(B)
            loss      = ce + cfg.lambda_hint * hint_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.item())

        avg = running / max(1, len(loader))
        print(f"[HINT] epoch {ep+1}/{cfg.epochs} - avg loss: {avg:.4f}")

    model.eval()


def generate_gradcam_overlay_library(
    model: nn.Module, target_layer: nn.Module, image_file: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        overlay_bgr: uint8 BGR image
        cam_resized: float CAM [H,W] in [0,1]
    """
    inp, rgb_np = load_image_tensor(image_file)
    with torch.no_grad():
        logits   = model(inp)
        pred_cls = int(torch.argmax(logits, dim=1).item())

    targets  = [ClassifierOutputTarget(pred_cls)]
    cam      = GradCAM(model=model, target_layers=[target_layer])
    cam_map  = cam(input_tensor=inp, targets=targets)[0]

    cam_resized = cv2.resize(cam_map, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    overlay_rgb = show_cam_on_image(rgb_np, cam_resized, use_rgb=True)
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    return overlay_bgr, cam_resized


def get_random_file(directory_path: str, want_dir: bool = False,
                    dir_like: Optional[str] = None) -> Optional[str]:
    """Pick a random file (or directory) from a path"""
    try:
        entries = os.listdir(directory_path)
        if not want_dir:
            files = [e for e in entries if os.path.isfile(os.path.join(directory_path, e))]
        else:
            files = [e for e in entries
                     if os.path.isdir(os.path.join(directory_path, e))
                     and (dir_like in e if dir_like else True)]
        if not files:
            return None
        return os.path.join(directory_path, random.choice(files))
    except Exception:
        return None


class SimpleChartDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels      = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        p      = self.image_paths[idx]
        y      = int(self.labels[idx])
        inp, _ = load_image_tensor(p)
        if inp.ndim == 4 and inp.shape[0] == 1:
            inp = inp.squeeze(0)
        return inp, torch.tensor(y, dtype=torch.long)


# ----------------------------
# Main
# ----------------------------

def extract_index(filename):
    # Extracts trailing integer from filenames like
    # 'line_chart_color_with_label_42.png' -> 42
    match = re.search(r'_(\d+)\.(png|jpg|jpeg)$', filename, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def main():
    #do_hint_finetune
    variants = [
        ("occlude_white", "occlude", 1.0),
    ]

    for dataset_name in DATASET_NAMES:
        for encoding_type in ENCODING_TYPES:
            print(f"\n=== Dataset={dataset_name}  Encoding={encoding_type} ===")

            try:
                # ----------------------------------------------------------
                # get num_classes from the .ts file
                # ----------------------------------------------------------
                ts_path = TS_TRAIN_PATHS.get(dataset_name)
                if ts_path is None or not os.path.exists(ts_path):
                    raise FileNotFoundError(f".ts file not found: {ts_path}")

                labels_all  = load_ts_labels(ts_path)
                num_classes = len(set(labels_all))
                print(f"    num_classes={num_classes}  (from {os.path.basename(ts_path)})")
                print(f"    model_path={MODEL_PATHS[dataset_name][encoding_type]}")
                print(f"    exists={os.path.exists(MODEL_PATHS[dataset_name][encoding_type])}")

                model, target_layer = load_model_for_encoding(
                    dataset_name, encoding_type, num_classes
                )

                # ----------------------------------------------------------
                # HINT fine-tuning process
                # ----------------------------------------------------------
                if do_hint_finetune:
                    train_dir = os.path.join(
                        IMAGE_ROOT,
                        f"{dataset_name}_images",
                        f"{encoding_type}{chart_config[encoding_type]}",
                        "train",
                    )

                    if os.path.exists(train_dir):
                        image_paths = sorted(
                            [
                                os.path.join(train_dir, f)
                                for f in os.listdir(train_dir)
                                if f.lower().endswith((".png", ".jpg", ".jpeg"))
                            ],
                            key=lambda p: extract_index(os.path.basename(p))
                        )

                        #  _occlude_white.png masks
                        print(f"[HINT] Generating Grad-CAM masks for {len(image_paths)} training images...")
                        for img_path in image_paths:
                            base   = os.path.splitext(os.path.basename(img_path))[0]
                            prefix = f"{dataset_name}_{encoding_type}_{base}"

                            # Skip already-generated files
                            if os.path.exists(os.path.join(OUTPUT_DIR, f"{prefix}_occlude_white.png")):
                                continue

                            overlay_bgr, cam_resized = generate_gradcam_overlay_library(
                                model, target_layer, img_path
                            )
                            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{prefix}_gradcam.png"), overlay_bgr)

                            _, rgb_np = load_image_tensor(img_path)
                            bbox = cam_to_bbox(cam_resized, topk_frac=BOX_TOPK_FRAC)
                            for tag, mode, fill in variants:
                                masked  = apply_box_mask(rgb_np, bbox, mode=mode, fill_value=fill)
                                out_bgr = cv2.cvtColor((masked * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                                cv2.imwrite(os.path.join(OUTPUT_DIR, f"{prefix}_{tag}.png"), out_bgr)

                        print(f"[HINT] Mask generation done → {OUTPUT_DIR}")

                        # build dataset with actual labels
                        if len(labels_all) != len(image_paths):
                            raise RuntimeError(
                                f"Label count ({len(labels_all)}) != image count ({len(image_paths)}) "
                                f"for {dataset_name}/{encoding_type}. "
                                "Check that the .ts file and image folder are the same split."
                            )

                        ds = ChartWithAttentionDataset(
                            image_paths, labels_all,
                            ATTENTION_ROOT,
                            dataset_name=dataset_name,
                            encoding_type=encoding_type,
                        )
                        dl = DataLoader(ds, batch_size=hint_cfg.batch_size,
                                        shuffle=True, num_workers=0)

                        print(f"[HINT] Fine-tuning on {len(ds)} images...")
                        hint_finetune(model, target_layer, dl, hint_cfg)

                        # save HINT-updated model
                        baseline_path  = MODEL_PATHS[dataset_name][encoding_type]
                        hint_save_path = os.path.join(
                            os.path.dirname(baseline_path), "hint_finetuned_model.pth"
                        )
                        torch.save(model.state_dict(), hint_save_path)
                        print(f"[HINT] Updated model saved → {hint_save_path}")

                    else:
                        print(f"[HINT] Train dir not found, skipping: {train_dir}")

                # ----------------------------------------------------------
                # record: generate overlay + masked variant for one random test image
                # ----------------------------------------------------------
                test_dir = os.path.join(
                    IMAGE_ROOT,
                    f"{dataset_name}_images",
                    f"{encoding_type}{chart_config[encoding_type]}",
                    "test",
                )
                
                if not os.path.exists(test_dir):
                    root_dir = os.path.join(IMAGE_ROOT, f"{dataset_name}_images")
                    alt = get_random_file(root_dir, want_dir=True, dir_like=encoding_type)
                    if alt:
                        test_dir = os.path.join(alt, "test")

                img_path = get_random_file(test_dir, want_dir=False)
                if img_path is None or not os.path.exists(img_path):
                    print(f"No test image found under {test_dir}. Skipping.")
                    continue

                overlay_bgr, cam_resized = generate_gradcam_overlay_library(
                    model, target_layer, img_path
                )
                base         = os.path.splitext(os.path.basename(img_path))[0]
                overlay_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_{encoding_type}_{base}_gradcam.png")
                cv2.imwrite(overlay_path, overlay_bgr)
                print(f"Saved overlay: {overlay_path}")

                _, rgb_np = load_image_tensor(img_path)
                bbox = cam_to_bbox(cam_resized, topk_frac=0.15)
                for tag, mode, fill in variants:
                    masked  = apply_box_mask(rgb_np, bbox, mode=mode, fill_value=fill)
                    out_bgr = cv2.cvtColor((masked * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(
                        os.path.join(OUTPUT_DIR, f"{dataset_name}_{encoding_type}_{base}_{tag}.png"),
                        out_bgr,
                    )
                print(f"Saved masked variants for: {base}")

            except Exception as e:
                traceback.print_exc()
                print(f"[SKIP] {dataset_name}/{encoding_type} — {e}")
                continue

    print(f"\nDone. Outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()