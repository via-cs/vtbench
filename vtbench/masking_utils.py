"""
Utilities for generating a bounding-box mask from a Grad-CAM heatmap and
applying simple white boxing
"""

from typing import Tuple
import numpy as np


def cam_to_bbox(cam: np.ndarray, topk_frac: float = 0.15) -> Tuple[int, int, int, int]:
    """Convert a CAM heatmap into a bounding box around the top-k fraction of pixels

    Args:
        cam: HxW float array in [0, 1]
        topk_frac: fraction of pixels considered "important"

    Returns:
        (x1, y1, x2, y2) bbox where x2/y2 are exclusive.
    """
    h, w = cam.shape
    flat = cam.reshape(-1)
    k = max(1, int(topk_frac * flat.size))

    # threshold = k-th largest value
    thresh = np.partition(flat, -k)[-k]
    mask = (cam >= thresh)

    ys, xs = np.where(mask)
    if xs.size == 0:
        # fallback: whole image
        return 0, 0, w, h

    x1, x2 = int(xs.min()), int(xs.max() + 1)
    y1, y2 = int(ys.min()), int(ys.max() + 1)
    return x1, y1, x2, y2


def apply_box_mask(
    rgb: np.ndarray,
    bbox: Tuple[int, int, int, int],
    mode: str = "occlude",
    fill_value: float = 0.0,
) -> np.ndarray:
    """Apply a box mask to an RGB image.

    Args:
        rgb: HxWx3 float array in [0,1]
        bbox: (x1,y1,x2,y2) with x2/y2 exclusive
        mode:
            - "occlude": mask the bbox region (removes important region)
            - "keep": mask everything outside the bbox (keeps only important region)
        fill_value:1.0 for white

    Returns:
        Masked HxWx3 float array in [0,1]
    """
    x1, y1, x2, y2 = bbox
    out = rgb.copy()

    if mode == "occlude":
        out[y1:y2, x1:x2, :] = fill_value
    elif mode == "keep":
        out[:] = fill_value
        out[y1:y2, x1:x2, :] = rgb[y1:y2, x1:x2, :]
    else:
        raise ValueError("mode must be 'occlude' or 'keep'")

    return out