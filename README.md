# VEIL
<img width="3527" height="605" alt="image" src="https://github.com/user-attachments/assets/d4873af8-3bde-459a-9381-9905ef08dd38" />

This repository contains the VEIL research project for chart-based and multimodal time-series classification.

The VEIL implementation lives under `vtbench/VEIL/VEIL`, and it is designed to train and evaluate models using chart image encodings, raw numerical inputs, or both.

## What VEIL does

- Converts time-series data into visual chart modalities
- Trains chart-based CNN models and multimodal fusion architectures
- Evaluates classification performance across different modality setups
- Supports configurable experiment definitions through YAML files
- Outputs results to configurable directories for downstream analysis

## VEIL folder structure

- `vtbench/VEIL/VEIL/main.py` — primary entry point for running experiments
- `vtbench/VEIL/VEIL/config/` — example YAML configuration files
- `vtbench/VEIL/VEIL/data/` — data loading utilities and dataset helpers
- `vtbench/VEIL/VEIL/train/` — training and evaluation pipelines
- `vtbench/VEIL/VEIL/models/` — model definitions for chart and numerical branches
- `vtbench/VEIL/VEIL/esi_visualization/` — visualization utilities for interpretability
- `vtbench/VEIL/VEIL/results/` — default output location for experiment results
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

```bash
python vtbench/VEIL/VEIL/main.py --config vtbench/VEIL/VEIL/config/single_modal_chart.yaml
```

Or change into the VEIL working directory:

```bash
cd vtbench/VEIL/VEIL
python main.py --config config/single_modal_chart.yaml
```

## Configuring experiments

Example config files are stored in `vtbench/VEIL/VEIL/config/`:

- `single_modal_chart.yaml` — single chart image classification
- `two_branch.yaml` — chart image + raw numerical fusion
- `multi_modal_chart.yaml` — multiple chart image modalities

Each config defines:

- dataset paths and names
- image generation behavior
- model type and branch settings
- training hyperparameters
- output directory and model saving options

### Example config fields

- `dataset.train_path` / `dataset.test_path` — UCR-style `.ts` dataset paths
- `image_generation.generate_images` — whether to generate chart images
- `model.type` — experiment type such as `single_modal_chart` or `two_branch`
- `output.dir` — where results and checkpoints are saved

## Adding your own dataset

1. Place your dataset in UCR/UEA-style `.ts` format.
2. Update a YAML config file with the correct `dataset.train_path` and `dataset.test_path`.
3. Set `image_generation.generate_images: true` to create chart images.
4. Run the VEIL script.

## Output and results

VEIL saves experiment outputs under the configured `output.dir` in the YAML file, for example:

```yaml
output:
  dir: results/single_chart/
  save_model: true
```

When run from `vtbench/VEIL/VEIL`, this creates `vtbench/VEIL/VEIL/results/single_chart/`.

## Notes

- Use absolute or project-relative dataset paths in the YAML config.
- Reuse generated images by setting `overwrite_existing: false`.
- Extend the code by adding new model branches in `vtbench/VEIL/VEIL/models/` and new training logic in `vtbench/VEIL/VEIL/train/`.

## Quick start

```bash
cd vtbench/VEIL
pip install -r requirements.txt
cd VEIL
python main.py --config config/single_modal_chart.yaml
```

This README now targets the VEIL subproject and its folder structure explicitly.
