from __future__ import print_function
import os
# from train_simsiam_heat import train
import numpy as np
from skimage import color
from PIL import Image

import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from NCE.NCECriterion import CrossEntropyLoss_label_smooth
from torchvision import transforms


class ImageRealFolderInstance(Dataset):
    def __init__(self, split, transform=None, target_transform=None, return_idx=False):
        root = './mini_imagenet/'
        train_data = np.load(os.path.join(root, 'miniimagenet_train_filePath.npy'))
        self.img = train_data.reshape(-1)
        self.transform = transform
        self.label = []
        self.return_idx = return_idx
        
        for i in range(train_data.shape[0]):
            self.label.extend([i for _ in range(len(train_data[i]))])
        self.label = np.array(self.label)
        print(self.img.shape)
        print(self.label.shape)
    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        img = openImage(self.img[idx])
        img = self.transform(img)
        y = self.label[idx]
        return idx, img, y


class MiniTestDataset(Dataset):
    def __init__(self, n_way, k_shot, k_query, task_num, imgsz=84):
        root = './mini_imagenet/'
        test_data = np.load(os.path.join(root, 'miniimagenet_test_filePath.npy'))
        self.data = test_data
        self.resize = imgsz
        self.n_cls = self.data.shape[0] 
        self.n_way = n_way  # n way
        self.k_shot = k_shot  # k shot
        self.k_query = k_query  # k query
        self.task_num = task_num
        image_size = imgsz
        crop_padding = 8
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)


        self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                            transforms.Resize(image_size + crop_padding),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                            normalize,
                        ])

        # random shuffle data
        self.task = []
        for _ in range(self.task_num):
            x_spt, y_spt, x_qry, y_qry = [], [], [], []
            selected_cls = np.random.choice(self.data.shape[0], self.n_way, False)
            temp_df_img = []
            for j, cur_class in enumerate(selected_cls):

                selected_img = np.random.choice(600, self.k_shot + self.k_query, False)
                

                x_spt.append(self.data[cur_class][selected_img[:self.k_shot]])
                x_qry.append(self.data[cur_class][selected_img[self.k_shot:]])
                y_spt.append([j for _ in range(self.k_shot)]) 
                y_qry.append([j for _ in range(self.k_query)])

            # shuffle inside a batch
            perm = np.random.permutation(self.n_way * self.k_shot)
            x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot)[perm]
            y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]

            perm = np.random.permutation(self.n_way * self.k_query)
            x_qry = np.array(x_qry).reshape(self.n_way * self.k_query)[perm]
            y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]

            
            self.task.append([x_spt, y_spt, x_qry, y_qry])

        
    def __len__(self):
        return len(self.task) 
    
    def __getitem__(self, index):
        task = self.task[index]
        x_spt, y_spt, x_qry, y_qry = task
        x_spt_img, x_qry_img = torch.zeros((self.n_way * self.k_shot, 3, self.resize, self.resize)), torch.zeros((self.n_way * self.k_query, 3, self.resize, self.resize))
        
        for i in range(len(x_spt)):
            x_spt_img[i] = self.transform(x_spt[i]).float()

        for i in range(len(x_qry)):
            x_qry_img[i] = self.transform(x_qry[i]).float()
        
        x_spt_img = x_spt_img.reshape(self.n_way * self.k_shot, 3, self.resize, self.resize)
        x_qry_img = x_qry_img.reshape(self.n_way * self.k_query, 3, self.resize, self.resize)
        return [x_spt_img, y_spt, x_qry_img, y_qry]

def openImage(x):
    return Image.open(x).convert('RGB')



class RGB2Lab(object):
    """Convert RGB PIL image to ndarray Lab."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2lab(img)
        return img


class RGB2HSV(object):
    """Convert RGB PIL image to ndarray HSV."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2hsv(img)
        return img


class RGB2HED(object):
    """Convert RGB PIL image to ndarray HED."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2hed(img)
        return img


class RGB2LUV(object):
    """Convert RGB PIL image to ndarray LUV."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2luv(img)
        return img


class RGB2YUV(object):
    """Convert RGB PIL image to ndarray YUV."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2yuv(img)
        return img


class RGB2XYZ(object):
    """Convert RGB PIL image to ndarray XYZ."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2xyz(img)
        return img


class RGB2YCbCr(object):
    """Convert RGB PIL image to ndarray YCbCr."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ycbcr(img)
        return img


class RGB2YDbDr(object):
    """Convert RGB PIL image to ndarray YDbDr."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ydbdr(img)
        return img


class RGB2YPbPr(object):
    """Convert RGB PIL image to ndarray YPbPr."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ypbpr(img)
        return img


class RGB2YIQ(object):
    """Convert RGB PIL image to ndarray YIQ."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2yiq(img)
        return img


class RGB2CIERGB(object):
    """Convert RGB PIL image to ndarray RGBCIE."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2rgbcie(img)
        return img
