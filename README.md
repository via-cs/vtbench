# VEIL
<img width="3527" height="605" alt="image" src="https://github.com/user-attachments/assets/d4873af8-3bde-459a-9381-9905ef08dd38" />

VEIL is a diagnostic framework for detecting and mitigating visual encoding hijacking in image-based time series classification.

## What VEIL does (source code)
- Converts time-series data into visual chart modalities
- Trains single chart chart-based CNN models
- Supports configurable experiment definitions through YAML files
- Outputs results to configurable directories for downstream analysis
- Evaluates classification performance between a finetuned and baseline model

## VEIL folder structure

- `vtbench/VEIL/VEIL/generate_configs_and_run.py` — primary entry point before running experiments
- `vtbench/VEIL/VEIL/config/generated_configs` — YAML configuration files
- `vtbench/VEIL/VEIL/data/` — data loading utilities and dataset helpers
- `vtbench/VEIL/VEIL/train/` — training and evaluation pipelines
- `vtbench/VEIL/VEIL/models/` — model definitions for chart and numerical branches
- `vtbench/VEIL/VEIL/{dataset}_results/` — default output location for experiment results
- `vtbench/VEIL/requirements.txt` — dependency list for running VEIL

## Requirements

Install the VEIL dependencies:

```bash
cd vtbench/VEIL
pip install -r requirements.txt
```

Required packages include:

- Python 3.8+
- torch >= 1.9
- torchvision >= 0.10
- numpy
- scikit-learn
- imbalanced-learn
- PyYAML
- matplotlib
- pandas
- seaborn
- umap-learn
- opencv-python
- grad-cam

## Running VEIL

From the project root, use the VEIL main script with a config file:

- to create a specific YAML for a singular chart encoding (for a singular dataset)
```bash
python vtbench/VEIL/VEIL/main.py --config vtbench/VEIL/VEIL/config/single_modal_chart.yaml
```
- to automate the creation of YAML files for all 4 chart encodings across all 31 UCR datasets
[first traverse into vtbench/VEIL/VEIL]
```bash
python generate_configs_and_run.py --config vtbench/VEIL/VEIL/config/single_modal_chart.yaml
```

## Configuring experiments

Example config files are stored in `vtbench/VEIL/VEIL/config/generated_config`:

- `single_modal_chart.yaml` — single chart image classification

Each config defines:

- dataset paths and names
- image generation behavior
- model type and branch settings
- training hyperparameters
- output directory and model saving options

### Config fields

- `dataset.train_path` / `dataset.test_path` — UCR-style `.ts` dataset paths
- `image_generation.generate_images` — whether to generate chart images
- `model.type` — experiment type such as `single_modal_chart`
- `model.num_classes`- number of classes for a given dataset
- `chart_branches.branch_1.chart_type:`- type of chart encoding
- `chart_branches.branch_1.label_mode` - labeled or unlabeled images
- `training.batch_size`- number of training samples per epoch
- `training.epochs`- number of passes through training dataset
- `output.dir` — where results and checkpoints are saved

## Image generation
1. Update a YAML config file with `image_generation.generate_images: true` to create chart images.

## Output and results

VEIL saves experiment outputs under the configured `output.dir` in the YAML file, for example:

```yaml
output:
  dir: Adiac_results/single_chart/line/
  save_model: true
```

## Notes

- Use absolute or project-relative dataset paths in the YAML config.
- Reuse generated images by setting `image_generation.overwrite_existing: false` in the config YAML files.

## Quick start

```bash
cd vtbench/VEIL
pip install -r requirements.txt
cd VEIL
python generate_configs_and_run.py
```

## USE
- Compare chart encodings – Train identical CNN backbones on line, area, bar, and scatter renditions of the same time series to measure how encoding choice affects representations.
- Detect encoding hijacking – Compute Encoding Sensitivity Indices (ESI_CKA, ESI_PROBE) to quantify whether models rely on temporal structure or visual artifacts.
- Test cross-encoding transfer – Use linear probing across chart types to see if features learned from one encoding generalize to another.
- Visualize model attention – Generate Grad-CAM heatmaps to identify whether models focus on signal patterns, encoding-specific cues, or chart artifacts.
- Quantify visual cue reliance – Apply controlled perturbations (blur, bar merging, alpha fading) and measure accuracy drops to diagnose encoding dependence.
- Mitigate encoding bias – Run HINT-based attention guidance to redirect model focus toward semantically meaningful temporal regions
- Analyze representations – Measure alignment between feature spaces using Centered Kernel Alignment (CKA), assess intrinsic dimensionality via PCA, and visualize class versus encoding clustering patterns with UMAP.
