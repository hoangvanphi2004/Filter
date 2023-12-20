import torch;
import config;
from Base import config as configBase
import albumentations as A
from albumentations.pytorch import ToTensorV2;
from albumentations.augmentations import geometric, crops;

def imageAug(image):
    transform = A.Compose([
        crops.transforms.Crop(x_min = config.PADDING_WIDTH, y_min = 0, x_max = config.PADDING_WIDTH + config.INPUT_IMAGE_HEIGHT_SIZE, y_max = config.INPUT_IMAGE_HEIGHT_SIZE),
        geometric.resize.SmallestMaxSize(configBase.IMG_SIZE),
        A.transforms.Normalize(),
        ToTensorV2()
    ]);
    image = transform(image = image)['image'];
    return image;