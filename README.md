# VTBench
VTBench is a Python package for time-series classification using visual chart representations. It supports multimodal architectures that combine image-based chart features with numerical time-series features for robust classification.

## Core Functional Modules

### `vtbench/data/` – Data Pipeline

- **`loader.py`**: Parses UCR `.ts` format, creates stratified splits, dataloaders  
- **`chart_generator.py`**: Converts time-series to images (line, bar, area, scatter)  
- **Responsibilities**: Data ingestion, preprocessing, and image generation

---

### `vtbench/models/` – Model Definitions

- **`chart_models/`**: CNNs for chart image classification (SimpleCNN, DeepCNN)  
- **`numerical/`**: Models for raw numerical input (FCN, OSCNN, Transformer)  
- **`multimodal/`**: Fusion networks combining image and numerical features  
- **Responsibilities**: Feature extraction and model definition

---

### `vtbench/train/` – Training Pipeline

- **`trainer.py`**: Manages the training loop for all model types  
- **`evaluate.py`**: Runs evaluation and metrics logging  
- **`factory.py`**: Dynamically builds models from config  


# Usage

Install the package in editable mode with modern packaging:

```bash 
pip install -e .
```

Then run your experiment using:

```bash 
vtbench --config vtbench/config/example_config.yaml
```

This command will:

- Parse the YAML config
- Generate required chart images from the time-series data
- Load data and initialize the appropriate model
- Train and evaluate the model
- Save results to the results/ folder


# Configuration Files

VTBench is driven by YAML configuration files that define model type, data paths, training parameters, fusion strategies and more. You can run VTBench with any of the following configs using:

```bash
vtbench --config vtbench/config/<config_file>.yaml
```

## Available Configs

###  `single_modal_chart.yaml`
Runs a single chart-type image model (e.g., line, bar, etc.) using a CNN. Useful for testing how well visual representations alone perform on classification tasks.

### `two_branch.yaml`
Combines chart image input with raw numerical data in a two-branch architecture. Helps compare unimodal vs. multimodal performance with a simple fusion strategy.

###  `multi_modal_chart.yaml`
Uses multiple chart types (e.g., line, bar, scatter, area) in parallel branches. Demonstrates how different visual representations can complement each other in a multimodal structure.
