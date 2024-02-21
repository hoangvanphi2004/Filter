import torch;
from torch.utils.data import Dataset, DataLoader, random_split;
from torchvision.transforms import v2
from torchvision.transforms import ToTensor, Resize, Normalize, Compose;
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2;
from albumentations.augmentations import geometric, crops;
import glob;
import pytorch_lightning as pl;
from PIL import Image;
import numpy as np;
import re;
import math;
import random;
from config import IMG_SIZE;
import xml.etree.ElementTree as ET
import random;

class Data(Dataset):
    def dataAugment(self, img, label):
        # ### ------------ Flip Image ---------------- ###
        # labelImg = torch.zeros(1, IMG_SIZE, IMG_SIZE);
        # for x, y in label:
        #     labelImg[0, int(y), int(x)] = 255;
        # isHFlip = (random.random() > 0.5);
        # isVFlip = (random.random() > 0.5);
        # if(isHFlip):
        #     img = TF.hflip(img);
        #     labelImg = TF.hflip(labelImg);
        # if(isVFlip):
        #     img = TF.hflip(img);
        #     labelImg = TF.hflip(labelImg);
        # label = (labelImg == 255).nonzero(as_tuple = False)[:, 1:];
        
        # ### ---- Resize Image to IMG_SIZE, follow by label with normal transform ---- ###
        # width = img.size()[2];
        # height = img.size()[1];
        # if(width <= height):
        #     resizeImgSize = (int(IMG_SIZE / width * height), int(IMG_SIZE));
        #     ratio = IMG_SIZE / width
        # else:
        #     resizeImgSize = (int(IMG_SIZE), int(IMG_SIZE / height * width));
        #     ratio = IMG_SIZE / height;

        # for index in range(len(label['box'])):
        #     label['box'][index] = label['box'][index] * ratio;
        # label['points'] = label['points'] * ratio; 
        
        # img = Resize(size = resizeImgSize, antialias=True)(img);
        
        # if(width <= height):
        #     top = max(0, min(int(label['box'][1] - (IMG_SIZE - label['box'][3]) / 2), int(IMG_SIZE / width * height) - IMG_SIZE));
        #     left = 0;
        #     img = TF.crop(img, top, left, IMG_SIZE, IMG_SIZE);
        #     label['box'][1] = label['box'][1] - top; 
        #     label['points'][:, 1] = label['points'][:, 1] - top; 
        # else:
        #     top = 0;
        #     left = max(0, min(int(label['box'][0] - (IMG_SIZE - label['box'][2]) / 2), int(IMG_SIZE / height * width) - IMG_SIZE));
        #     img = TF.crop(img, top, left, IMG_SIZE, IMG_SIZE);
        #     label['box'][0] = label['box'][0] - left;  
        #     label['points'][:, 0] = label['points'][:, 0] - left; 
        
        # ### --------------- Image Aug Without YOLO --------------- ###
        # width = img.shape[1];
        # height = img.shape[0];    
        # label['box'] = np.array([label['box']]);
        
        # transform = A.Compose([\
        #     geometric.resize.SmallestMaxSize(max_size = IMG_SIZE),\
        # ], bbox_params = A.BboxParams(format = "coco", label_fields = ["class_labels"]), keypoint_params = A.KeypointParams(format = "xy", remove_invisible=False));
        # transformResult = transform(image = img, bboxes = label["box"], class_labels = ["human-face"], keypoints = label["points"]);
        
        # img = transformResult['image'];
        # label['box'] = transformResult['bboxes'];
        # label['points'] = transformResult['keypoints'];
        
        # if(width <= height):
        #     top = max(0, min(int(label['box'][0][1] - (IMG_SIZE - label['box'][0][3]) / 2), int(IMG_SIZE / width * height) - IMG_SIZE));
        #     left = 0;
        # else:
        #     top = 0;
        #     left = max(0, min(int(label['box'][0][0] - (IMG_SIZE - label['box'][0][2]) / 2), int(IMG_SIZE / height * width) - IMG_SIZE));
        
        # transform = A.Compose([\
        #     crops.transforms.Crop(x_min = left, y_min = top, x_max = left + IMG_SIZE, y_max = top + IMG_SIZE),\
        #     geometric.transforms.ShiftScaleRotate(scale_limit = (-0.4, 0)),\
        #     A.transforms.Normalize(),\
        #     ToTensorV2()\
        # ], bbox_params = A.BboxParams(format = "coco", label_fields = ["class_labels"]), keypoint_params = A.KeypointParams(format = "xy", remove_invisible=False));
        # transformResult = transform(image = img, bboxes = label["box"], class_labels = ["human-face"], keypoints = label["points"]); 
        
        img = img.crop((label['box'][0], label['box'][1], label['box'][0] + label['box'][2], label['box'][1] + label['box'][3]));
        label['points'][:, 0] -= label['box'][0];
        label['points'][:, 1] -= label['box'][1];
        
        if(len(img.size) != 3):
            img = img.convert('L');
            img = Image.merge('RGB', [img] * 3);
        img = np.array(img);
        
        width = img.shape[1];
        height = img.shape[0];    
        label['box'] = np.array([label['box']]);
        
        transform = A.Compose([\
            geometric.resize.Resize(IMG_SIZE, IMG_SIZE, always_apply = True),\
            geometric.transforms.ShiftScaleRotate(),\
            A.transforms.Normalize(),\
            ToTensorV2()\
        ], keypoint_params = A.KeypointParams(format = "xy", remove_invisible=False));
        transformResult = transform(image = img, class_labels = ["human-face"], keypoints = label["points"]); 
        
        img = transformResult['image'];
        label['points'] = torch.tensor(transformResult['keypoints']);
        
        return img, label;
        
    def __init__(self):
        data = ET.parse('cs194-26-fa20-proj4/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml');
        self.data = data.getroot().find('images').findall('image');
    def __len__(self):
        return len(self.data);
    def __getitem__(self, index):
        image = Image.open('cs194-26-fa20-proj4/ibug_300W_large_face_landmark_dataset/' + self.data[index].attrib['file']);
        
        box = self.data[index].find('box');
        points = np.array([[int(point.attrib['x']), int(point.attrib['y'])] for point in box.findall('part')]);
        label = {};
        
        # ### --------------- Image Aug Without YOLO --------------- ###
        
        # if(len(img.size) != 3):
        #     img = img.convert('L');
        #     img = Image.merge('RGB', [img] * 3);
        # img = np.array(img);
        
        # label['box'] = np.array([max(int(box.attrib['left']), 0),\
        #                             max(int(box.attrib['top']), 0),\
        #                             min(int(box.attrib['width']), image.shape[1] - max(int(box.attrib['left']), 0) - 1),\
        #                             min(int(box.attrib['height']), image.shape[0] - max(int(box.attrib['top']), 0) - 1)]);
        
        label['box'] = np.array([int(box.attrib['left']),\
                                int(box.attrib['top']),\
                                int(box.attrib['width']),\
                                int(box.attrib['height']),]);
        
        label['points'] = points;
        
        image, label = self.dataAugment(image, label);
        label['points'] = torch.stack([torch.stack([point[0], point[1]], dim = 0) for point in label['points']], dim = 0)
        
        return image, label;
            

class DataModule(pl.LightningDataModule):
    def customCol(self, batch):
        image = torch.stack([item[0] for item in batch], dim = 0);
        box = [item[1]['box'] for item in batch];
        points = torch.stack([item[1]['points'] for item in batch], dim = 0);
        return [image, box, points];

    def __init__(self, batch_size, num_workers):
        super().__init__();
        self.batch_size = batch_size;
        self.num_workers = num_workers;
    def setup(self, stage):
        self.entireData = Data();
        self.trainData, self.validateData = random_split(self.entireData, [0.9, 0.1]);
    def train_dataloader(self):
        return DataLoader(self.trainData, batch_size = self.batch_size, num_workers = self.num_workers, shuffle = True, collate_fn = self.customCol);
    def val_dataloader(self):
        return DataLoader(self.validateData, batch_size = self.batch_size, num_workers = self.num_workers, shuffle = False, collate_fn = self.customCol);