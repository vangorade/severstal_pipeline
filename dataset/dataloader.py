# system
import os

# libraries
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# modules
from dataset import SteelDataset

# folder to load config file
CONFIG_PATH = "./"

# Function to load yaml configuration file


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


config = load_config("cfg.yaml")


def provider(
    data_folder,
    df_path,
    phase,
    mean=None,
    std=None,
    batch_size=config["batch_size"],
    num_workers=config["nworkers"],
):
    '''Returns dataloader for the model training'''
    df = pd.read_csv(df_path)
    df['ImageId_ClassId'] = df.ImageId.astype(
        str).str.cat(df.ClassId.astype(str), sep='_')
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
    df['defects'] = df.count(axis=1)

    train_df, val_df = train_test_split(
        df, test_size=config["validation_split"], stratify=df["defects"], random_state=69)
    df = train_df if phase == "train" else val_df
    image_dataset = SteelDataset(df, data_folder, mean, std, phase)
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )

    return dataloader
