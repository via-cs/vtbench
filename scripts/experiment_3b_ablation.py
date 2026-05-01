import argparse
import copy
import csv
import os

import numpy as np
import yaml
from torchvision import transforms
import matplotlib.pyplot as plt

from vtbench.train.trainer import train_model
from vtbench.train.evaluate import evaluate_model
from vtbench.data.loader import create_dataloaders
from vtbench.utils.ablation import apply_ablation, ablation_type_name, chart_dir_for


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _dataset_entries(exp_cfg):
    exp = exp_cfg["experiment"]
    dataset_root = exp.get("dataset_root", ".")
    dataset_ext = exp.get("dataset_ext", "tsv")
    entries = []
    for item in exp.get("datasets", []):
        if isinstance(item, str):
            name = item
            train_path = os.path.join(dataset_root, name, f"{name}_TRAIN.{dataset_ext}")
            test_path = os.path.join(dataset_root, name, f"{name}_TEST.{dataset_ext}")
            entries.append(
                {
                    "name": name,
                    "train_path": train_path,
                    "test_path": test_path,
                    "format": exp.get("dataset_format", "ucr"),
                }
            )
        else:
            name = item["name"]
            train_path = item.get(
                "train_path",
                os.path.join(dataset_root, name, f"{name}_TRAIN.{dataset_ext}"),
            )
            test_path = item.get(
                "test_path",
                os.path.join(dataset_root, name, f"{name}_TEST.{dataset_ext}"),
            )
            entries.append(
                {
                    "name": name,
                    "train_path": train_path,
                    "test_path": test_path,
                    "format": item.get("format", exp.get("dataset_format", "ucr")),
                }
            )
    return entries


def _build_run_config(exp_cfg, dataset_entry, chart_type):
    chart_defaults = exp_cfg.get("chart", {})
    bar_mode = chart_defaults.get("bar_mode", "border")
    scatter_mode = chart_defaults.get("scatter_mode", "plain")

    config = {
        "dataset": {
            "name": dataset_entry["name"],
            "train_path": dataset_entry["train_path"],
            "test_path": dataset_entry["test_path"],
            "format": dataset_entry.get("format", "ucr"),
        },
        "image_generation": copy.deepcopy(exp_cfg.get("image_generation", {})),
        "model": copy.deepcopy(exp_cfg.get("model", {})),
        "training": copy.deepcopy(exp_cfg.get("training", {})),
        "chart_branches": {
            "branch_1": {
                "chart_type": chart_type,
                "color_mode": chart_defaults.get("color_mode", "color"),
                "label_mode": chart_defaults.get("label_mode", "with_label"),
            }
        },
    }

    if chart_type == "bar":
        config["chart_branches"]["branch_1"]["bar_mode"] = bar_mode
    if chart_type == "scatter":
        config["chart_branches"]["branch_1"]["scatter_mode"] = scatter_mode

    return config


def _base_transforms(config):
    dataset_format = config["dataset"].get("format", "ucr")
    preserve_mv = bool(config.get("image_generation", {}).get("preserve_multivariate_size", True))
    chart_model = str(config["model"].get("chart_model", "")).lower()
    is_resnet18 = chart_model == "resnet18"
    pretrained = is_resnet18 and config["model"].get("pretrained", False)

    if dataset_format == "uea" and preserve_mv:
        base = [transforms.ToTensor()]
    else:
        base = [transforms.Resize((128, 128)), transforms.ToTensor()]

    if pretrained:
        base.append(
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        )

    return base


def _ablation_transform(chart_type, ablations_cfg):
    return lambda img: apply_ablation(img, chart_type, ablations_cfg)


def _set_chart_transform(chart_loaders, transform):
    if isinstance(chart_loaders, list):
        for loader in chart_loaders:
            loader.dataset.transform = transform
    else:
        chart_loaders.dataset.transform = transform


def _set_chart_dirs(chart_loaders, chart_branches, dataset_name, ablated_root, split):
    branch_cfgs = list(chart_branches.values())
    if isinstance(chart_loaders, list):
        if len(branch_cfgs) != len(chart_loaders):
            raise ValueError("Chart branch count does not match chart loaders.")
        for loader, cfg in zip(chart_loaders, branch_cfgs):
            chart_dir = chart_dir_for(
                ablated_root,
                dataset_name,
                cfg["chart_type"],
                cfg.get("color_mode", "color"),
                cfg.get("label_mode", "with_label"),
                split,
                bar_mode=cfg.get("bar_mode", "border"),
                scatter_mode=cfg.get("scatter_mode", "plain"),
            )
            if not os.path.isdir(chart_dir):
                raise FileNotFoundError(f"Ablated chart dir not found: {chart_dir}")
            loader.dataset.chart_dir = chart_dir
    else:
        cfg = branch_cfgs[0]
        chart_dir = chart_dir_for(
            ablated_root,
            dataset_name,
            cfg["chart_type"],
            cfg.get("color_mode", "color"),
            cfg.get("label_mode", "with_label"),
            split,
            bar_mode=cfg.get("bar_mode", "border"),
            scatter_mode=cfg.get("scatter_mode", "plain"),
        )
        if not os.path.isdir(chart_dir):
            raise FileNotFoundError(f"Ablated chart dir not found: {chart_dir}")
        chart_loaders.dataset.chart_dir = chart_dir


def _evaluate_with_transform(model, config, transform=None, ablated_root=None):
    loaders = create_dataloaders(config)
    if ablated_root:
        _set_chart_dirs(
            loaders["test"]["chart"],
            config["chart_branches"],
            config["dataset"]["name"],
            ablated_root,
            "test",
        )
    if transform is not None:
        _set_chart_transform(loaders["test"]["chart"], transform)

    model_type = config["model"]["type"]
    if model_type == "single_modal_chart":
        eval_type = "single_chart"
    elif model_type == "two_branch":
        eval_type = "two_branch"
    elif model_type == "multi_modal_chart":
        eval_type = "multi_chart"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return evaluate_model(
        model,
        loaders["test"]["chart"],
        loaders["test"]["numerical"],
        eval_type,
    )


def _plot_group(dataset_name, entries, group_types, title_suffix, output_path):
    ordered = [e for e in entries if e["chart_type"] in group_types]
    if not ordered:
        return

    labels = [e["ablation_type"] for e in ordered]
    deltas = [e["delta_accuracy"] for e in ordered]

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.bar(labels, deltas, color="#2f6d80")
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.set_ylabel("Delta accuracy")
    ax.set_xlabel("Ablation type")
    ax.set_title(f"{dataset_name} - {title_suffix}")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _interpret_dataset(entries):
    deltas = [e["delta_accuracy"] for e in entries]
    if not deltas:
        return "No ablation results available."
    mean_delta = float(np.mean(deltas))
    min_delta = float(np.min(deltas))
    if mean_delta <= -0.05 or min_delta <= -0.08:
        return (
            "Large drops after ablation; suggests reliance on encoding-specific details "
            f"(mean delta {mean_delta:.3f})."
        )
    if mean_delta <= -0.02:
        return (
            "Moderate sensitivity to ablation; fine-grained visual cues still matter "
            f"(mean delta {mean_delta:.3f})."
        )
    return (
        "Small change after ablation; performance likely driven by global shape "
        f"(mean delta {mean_delta:.3f})."
    )


def run_experiment(cfg):
    exp = cfg["experiment"]
    output_dir = exp.get("output_dir", "results/experiment_3b")
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    chart_types = exp.get("chart_types", ["line", "area", "bar", "scatter"])
    datasets = _dataset_entries(cfg)
    ablations_cfg = cfg.get("ablations", {})
    use_saved = bool(exp.get("use_saved_ablations", False))
    ablated_root = exp.get("ablated_image_root", "chart_images_ablated") if use_saved else None

    results = []

    for dataset_entry in datasets:
        dataset_name = dataset_entry["name"]
        for chart_type in chart_types:
            run_config = _build_run_config(cfg, dataset_entry, chart_type)
            train_config = copy.deepcopy(run_config)
            train_config["image_generation"].setdefault("generate_images", True)
            train_config["image_generation"].setdefault("overwrite_existing", False)

            model = train_model(train_config)

            eval_config = copy.deepcopy(run_config)
            eval_config["image_generation"]["generate_images"] = False
            eval_config["image_generation"]["overwrite_existing"] = False

            original = _evaluate_with_transform(model, eval_config)

            if use_saved:
                ablated = _evaluate_with_transform(model, eval_config, ablated_root=ablated_root)
            else:
                ablation_fn = _ablation_transform(chart_type, ablations_cfg)
                transform = transforms.Compose([ablation_fn] + _base_transforms(eval_config))
                ablated = _evaluate_with_transform(model, eval_config, transform=transform)

            original_acc = float(original["accuracy"])
            ablated_acc = float(ablated["accuracy"])
            delta = ablated_acc - original_acc

            results.append(
                {
                    "dataset": dataset_name,
                    "chart_type": chart_type,
                    "ablation_type": ablation_type_name(chart_type),
                    "original_accuracy": original_acc,
                    "ablated_accuracy": ablated_acc,
                    "delta_accuracy": delta,
                }
            )

    csv_path = os.path.join(output_dir, "ablation_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "chart_type",
                "ablation_type",
                "original_accuracy",
                "ablated_accuracy",
                "delta_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    md_path = os.path.join(output_dir, "ablation_results.md")
    with open(md_path, "w") as f:
        f.write("| dataset | encoding | ablation_type | original_acc | ablated_acc | delta |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for row in results:
            f.write(
                f"| {row['dataset']} | {row['chart_type']} | {row['ablation_type']} | "
                f"{row['original_accuracy']:.4f} | {row['ablated_accuracy']:.4f} | {row['delta_accuracy']:.4f} |\n"
            )

    interpretations_path = os.path.join(output_dir, "interpretations.txt")
    with open(interpretations_path, "w") as f:
        for dataset_entry in datasets:
            name = dataset_entry["name"]
            ds_entries = [r for r in results if r["dataset"] == name]
            f.write(f"{name}: {_interpret_dataset(ds_entries)}\n")

        f.write(
            "\nGuidance: small delta suggests encoding-invariant behavior; "
            "large negative delta suggests reliance on encoding-specific cues.\n"
        )

    for dataset_entry in datasets:
        name = dataset_entry["name"]
        ds_entries = [r for r in results if r["dataset"] == name]
        _plot_group(
            name,
            ds_entries,
            ["line", "area"],
            "Line/Area Ablations",
            os.path.join(plots_dir, f"{name}_line_area.png"),
        )
        _plot_group(
            name,
            ds_entries,
            ["bar", "scatter"],
            "Bar/Scatter Ablations",
            os.path.join(plots_dir, f"{name}_bar_scatter.png"),
        )

    print(f"Saved results to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Experiment 3B: Chart-Component Ablation")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
