import os
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

from models.chart_models.deepcnn import DeepCNN

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def cam_to_bbox(cam: np.ndarray, topk_frac: float = 0.15) -> Tuple[int, int, int, int]:
    """
    Convert a CAM heatmap into a bounding box by selecting the top-k fraction of pixels.

    Args:
        cam: HxW array in [0,1]
        topk_frac: fraction of pixels to consider "important"

    Returns:
        (x1, y1, x2, y2) bbox inclusive-exclusive in pixel coords.
    """
    h, w = cam.shape
    flat = cam.reshape(-1)
    k = max(1, int(topk_frac * flat.size))

    thresh = np.partition(flat, -k)[-k]
    mask = (cam >= thresh).astype(np.uint8)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        # fallback: whole image
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
    """
    Apply a box mask in one of four modes.

    Args:
        rgb_np: HxWx3 float in [0,1]
        bbox: (x1,y1,x2,y2)
        mode:
            - "keep": keep bbox region, mask everything else
            - "occlude": mask bbox region, keep everything else
        fill_value: 0.0 for black, 1.0 for white

    Returns:
        HxWx3 float image in [0,1]
    """
    x1, y1, x2, y2 = bbox
    out = rgb_np.copy()

    if mode == "keep":
        # mask outside the bbox
        out[:] = fill_value
        out[y1:y2, x1:x2, :] = rgb_np[y1:y2, x1:x2, :]
    elif mode == "occlude":
        # mask inside the bbox
        out[y1:y2, x1:x2, :] = fill_value
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return out
