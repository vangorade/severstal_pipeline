import yaml
import os
from segmentation_models_pytorch import UnetPlusPlus

# folder to load config file
CONFIG_PATH = "./"

# Function to load yaml configuration file


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


config = load_config("cfg.yaml")

model = UnetPlusPlus(encoder_name=config["encoder_name"],
                     encoder_weights=config["encoder_weights"],
                     classes=config["classes"])
