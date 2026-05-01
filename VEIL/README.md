# 📊 VEIL
*A Diagnostic Framework for Chart-Based Time-Series Classification*

- VTBench is a **Python package** for time-series classification using **visual chart representations** (line, bar, area, scatter) — individually or fused with mathematical encodings (GASF, MTF, RP, CWT, STFT).
- It provides a **modular, extensible framework** for training CNN classifiers and **diagnosing** what visual features they actually learn.

<img width="3527" height="605" alt="image" src="https://github.com/user-attachments/assets/94c22383-7414-4d6b-a99b-e68681ae0d30" />


## Highlights
- Auto-generate **4 chart types** from time series
- Compare **17 encodings**: 4 chart + 8 math + 2 RGB + 3 colormap
- Diagnostic toolkit: **CKA, linear probing, ESI, Grad-CAM, PCA, UMAP**
- **Chart perturbation** experiments to test encoding-specific reliance
- **HINT attention guidance** for redirecting model focus
- Built-in **UCR Archive** support (31 datasets)
- Config-driven & reproducible


## Installation
Install in editable mode:
```bash
git clone -b suranjana-code https://github.com/via-cs/vtbench.git
cd vtbench
pip install -e .
```


## Instructions to run

1. Create a folder named `data/` in the project root.
2. Place your time-series datasets in **.ts** format (UCR/UEA style) inside `data/`.
3. Run a YAML config:
```bash
   vtbench --config vtbench/config/<config_file>.yaml
```
4. On first run, chart images are **automatically generated** and saved under:
```bash
chart_images/<dataset_name>/...
```
Note: Images are reused on subsequent runs. To regenerate, set `generate_images` and `overwrite_existing` in your YAML.


## Configuration Files

VTBench is driven by YAML configs that define model type, data paths, training parameters, and analysis settings.

```bash
vtbench --config vtbench/config/<config_file>.yaml
```

This will:
- Parse the YAML config
- Generate required chart images
- Load data and initialize the model
- Train and evaluate
- Save results to `results/`


## Available Configs

#### `single_modal_chart.yaml`
Runs a single chart-type image model (e.g., line, bar) using a CNN. Useful for testing visual representations alone.


## Diagnostic experiments

Beyond standard training, VTBench includes scripts for diagnosing **what** the model is learning:

| Script | What it does |
|---|---|
| `extract_features_and_cka.py` | CKA representation similarity across chart types |
| `cross_enc_lin_probe.py` | Cross-encoding linear probing (transferability) |
| `pca.py` | Intrinsic dimensionality of learned features |
| `compute_esi.py` | Encoding Sensitivity Index (ESI_CKA, ESI_PROBE) |
| `gradcam_HINT_latest.py` | Grad-CAM attention overlays + HINT fine-tuning |
| `evaluate_hint.py` | Compare baseline vs HINT-finetuned accuracy |
| `experiment_3b_ablation.py` | Chart-component perturbation experiments |
| `run_supp6a.py` | Cross-family encoding comparison |

Run end-to-end orchestration:
```bash
python generate_configs_and_run.py
```
This auto-generates per-dataset configs and trains baselines for all chart types.


## Use
VTBench is designed for:
- **Diagnosing** what CNN-based time-series classifiers learn from visual encodings
- Benchmarking chart-based vs. numerical TSC
- Ablation studies on chart types, perturbations, and encoding families
- Reproducible research via config-driven design
- Extending to new encoders, encodings, and modalities
