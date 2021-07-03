# system

# libraries
from albumentations.pytorch import ToTensorV2
from albumentations import Compose
from albumentations.imgaug.transforms import Sharpen
from albumentations.augmentations.transforms import (
    HorizontalFlip,
    Normalize,
    RandomBrightnessContrast,
    CLAHE
)
# modules


def get_transforms(phase, mean, std):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                RandomBrightnessContrast(),
                HorizontalFlip(p=0.5),
                CLAHE(),
                Sharpen(),
            ]
        )
    list_transforms.extend(
        [
            Normalize(mean=mean, std=std, p=1),
            ToTensorV2(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms
