import os
from PIL import Image, ImageFilter
import numpy as np


def chart_dir_for(
    base_root,
    dataset_name,
    chart_type,
    color_mode,
    label_mode,
    split,
    bar_mode="border",
    scatter_mode="plain",
):
    base_dir = os.path.join(base_root, f"{dataset_name}_images")
    if chart_type == "area":
        return os.path.join(base_dir, f"area_charts_{color_mode}_{label_mode}", split)
    if chart_type == "bar":
        return os.path.join(base_dir, f"bar_charts_{bar_mode}_{color_mode}_{label_mode}", split)
    if chart_type == "line":
        return os.path.join(base_dir, f"line_charts_{color_mode}_{label_mode}", split)
    if chart_type == "scatter":
        return os.path.join(base_dir, f"scatter_charts_{scatter_mode}_{color_mode}_{label_mode}", split)
    raise ValueError(f"Unsupported chart type: {chart_type}")


def ablation_type_name(chart_type):
    if chart_type in ("line", "area"):
        return f"{chart_type}_smooth_blur"
    if chart_type == "bar":
        return "bar_merge_blur"
    if chart_type == "scatter":
        return "scatter_fade"
    return chart_type


def apply_ablation(img, chart_type, ablations_cfg):
    if chart_type in ("line", "area"):
        cfg = ablations_cfg.get("line_area", {})
        return _ablate_line_area(img, cfg)
    if chart_type == "bar":
        cfg = ablations_cfg.get("bar", {})
        return _ablate_bar(img, cfg)
    if chart_type == "scatter":
        cfg = ablations_cfg.get("scatter", {})
        return _ablate_scatter(img, cfg)
    raise ValueError(f"Unsupported chart type for ablation: {chart_type}")


def _nonwhite_mask(img, threshold):
    arr = np.asarray(img.convert("RGB"))
    mask = (arr < threshold).any(axis=2)
    return Image.fromarray((mask * 255).astype(np.uint8), mode="L")


def _blur_masked(img, mask, radius, expand):
    if expand and expand > 1:
        mask = mask.filter(ImageFilter.MaxFilter(expand))
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return Image.composite(blurred, img, mask)


def _ablate_line_area(img, cfg):
    mask = _nonwhite_mask(img, cfg.get("mask_threshold", 230))
    return _blur_masked(img, mask, cfg.get("blur_radius", 4), cfg.get("mask_expand", 3))


def _ablate_bar(img, cfg):
    scale = float(cfg.get("merge_scale", 0.7))
    blur_radius = float(cfg.get("blur_radius", 1.5))
    w, h = img.size
    w2 = max(1, int(w * scale))
    merged = img.resize((w2, h), resample=Image.BILINEAR).resize((w, h), resample=Image.BILINEAR)
    if blur_radius > 0:
        merged = merged.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return merged


def _ablate_scatter(img, cfg):
    mask = _nonwhite_mask(img, cfg.get("mask_threshold", 235))
    fade_alpha = float(cfg.get("fade_alpha", 0.2))
    fade_alpha = max(0.0, min(1.0, fade_alpha))
    white = Image.new("RGB", img.size, (255, 255, 255))
    faded = Image.blend(img, white, 1 - fade_alpha)
    return Image.composite(faded, img, mask)
