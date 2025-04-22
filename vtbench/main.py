# vtbench/main.py
import argparse
from vtbench.utils.config_parser import load_config
from vtbench.train.trainer import train_model
from vtbench.data.image_generator import generate_images_if_needed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    if config.get("preprocessing", {}).get("generate_images", True):
        generate_images_if_needed(config)

    train_model(config)

if __name__ == '__main__':
    main()
