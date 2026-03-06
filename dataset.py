"""
Dataset classes for few-shot learning.

Provides dataset loaders for training and testing on mini-ImageNet
and other few-shot learning benchmarks.
"""
from __future__ import print_function

import os

import numpy as np
import torch
from PIL import Image
from skimage import color
from torch.utils.data import Dataset
from torchvision import transforms


class ImageRealFolderInstance(Dataset):
    """
    Dataset for instance discrimination training.

    Loads images from mini-ImageNet training set with instance-level labels.
    """

    def __init__(self, split, transform=None, target_transform=None,
                 return_idx=False):
        """
        Initialize dataset.

        Args:
            split: Dataset split (train/val/test)
            transform: Image transformations
            target_transform: Target transformations
            return_idx: Whether to return image index
        """
        root = './mini_imagenet/'
        train_data = np.load(
            os.path.join(root, 'miniimagenet_train_filePath.npy')
        )
        self.img = train_data.reshape(-1)
        self.transform = transform
        self.label = []
        self.return_idx = return_idx

        # Create labels for each image
        for i in range(train_data.shape[0]):
            self.label.extend([i for _ in range(len(train_data[i]))])
        self.label = np.array(self.label)

        print("Images shape:", self.img.shape)
        print("Labels shape:", self.label.shape)

    def __len__(self):
        """Return dataset size."""
        return len(self.img)

    def __getitem__(self, idx):
        """
        Get item by index.

        Args:
            idx: Index

        Returns:
            Tuple of (index, image, label)
        """
        img = openImage(self.img[idx])
        img = self.transform(img)
        y = self.label[idx]
        return idx, img, y


class MiniTestDataset(Dataset):
    """
    Dataset for few-shot learning evaluation.

    Generates N-way K-shot tasks from mini-ImageNet test set.
    """

    def __init__(self, n_way, k_shot, k_query, task_num, imgsz=84):
        """
        Initialize test dataset.

        Args:
            n_way: Number of classes per task
            k_shot: Number of support samples per class
            k_query: Number of query samples per class
            task_num: Number of tasks to generate
            imgsz: Image size (default: 84)
        """
        root = './mini_imagenet/'
        test_data = np.load(
            os.path.join(root, 'miniimagenet_test_filePath.npy')
        )
        self.data = test_data
        self.resize = imgsz
        self.n_cls = self.data.shape[0]
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.task_num = task_num

        # Define image transformations
        image_size = imgsz
        crop_padding = 8
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)

        self.transform = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize(image_size + crop_padding),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])

        # Generate tasks
        self.task = []
        for _ in range(self.task_num):
            x_spt, y_spt, x_qry, y_qry = [], [], [], []
            selected_cls = np.random.choice(
                self.data.shape[0], self.n_way, False
            )

            for j, cur_class in enumerate(selected_cls):
                selected_img = np.random.choice(
                    600, self.k_shot + self.k_query, False
                )

                x_spt.append(self.data[cur_class][selected_img[:self.k_shot]])
                x_qry.append(self.data[cur_class][selected_img[self.k_shot:]])
                y_spt.append([j for _ in range(self.k_shot)])
                y_qry.append([j for _ in range(self.k_query)])

            # Shuffle within batch
            perm = np.random.permutation(self.n_way * self.k_shot)
            x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot)[perm]
            y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]

            perm = np.random.permutation(self.n_way * self.k_query)
            x_qry = np.array(x_qry).reshape(self.n_way * self.k_query)[perm]
            y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]

            self.task.append([x_spt, y_spt, x_qry, y_qry])

    def __len__(self):
        """Return number of tasks."""
        return len(self.task)

    def __getitem__(self, index):
        """
        Get task by index.

        Args:
            index: Task index

        Returns:
            List of [support_images, support_labels, query_images, query_labels]
        """
        task = self.task[index]
        x_spt, y_spt, x_qry, y_qry = task

        # Initialize tensors
        x_spt_img = torch.zeros(
            (self.n_way * self.k_shot, 3, self.resize, self.resize)
        )
        x_qry_img = torch.zeros(
            (self.n_way * self.k_query, 3, self.resize, self.resize)
        )

        # Load and transform images
        for i in range(len(x_spt)):
            x_spt_img[i] = self.transform(x_spt[i]).float()

        for i in range(len(x_qry)):
            x_qry_img[i] = self.transform(x_qry[i]).float()

        x_spt_img = x_spt_img.reshape(
            self.n_way * self.k_shot, 3, self.resize, self.resize
        )
        x_qry_img = x_qry_img.reshape(
            self.n_way * self.k_query, 3, self.resize, self.resize
        )
        return [x_spt_img, y_spt, x_qry_img, y_qry]


def openImage(x):
    """
    Open image from file path.

    Args:
        x: Image file path

    Returns:
        PIL Image in RGB format
    """
    return Image.open(x).convert('RGB')



# Color space conversion transforms (currently unused but kept for potential future use)

class RGB2Lab(object):
    """Convert RGB PIL image to Lab color space."""

    def __call__(self, img):
        """Convert image to Lab."""
        img = np.asarray(img, np.uint8)
        img = color.rgb2lab(img)
        return img


class RGB2HSV(object):
    """Convert RGB PIL image to HSV color space."""

    def __call__(self, img):
        """Convert image to HSV."""
        img = np.asarray(img, np.uint8)
        img = color.rgb2hsv(img)
        return img


class RGB2HED(object):
    """Convert RGB PIL image to HED color space."""

    def __call__(self, img):
        """Convert image to HED."""
        img = np.asarray(img, np.uint8)
        img = color.rgb2hed(img)
        return img


class RGB2LUV(object):
    """Convert RGB PIL image to LUV color space."""

    def __call__(self, img):
        """Convert image to LUV."""
        img = np.asarray(img, np.uint8)
        img = color.rgb2luv(img)
        return img


class RGB2YUV(object):
    """Convert RGB PIL image to YUV color space."""

    def __call__(self, img):
        """Convert image to YUV."""
        img = np.asarray(img, np.uint8)
        img = color.rgb2yuv(img)
        return img


class RGB2XYZ(object):
    """Convert RGB PIL image to XYZ color space."""

    def __call__(self, img):
        """Convert image to XYZ."""
        img = np.asarray(img, np.uint8)
        img = color.rgb2xyz(img)
        return img


class RGB2YCbCr(object):
    """Convert RGB PIL image to YCbCr color space."""

    def __call__(self, img):
        """Convert image to YCbCr."""
        img = np.asarray(img, np.uint8)
        img = color.rgb2ycbcr(img)
        return img


class RGB2YDbDr(object):
    """Convert RGB PIL image to YDbDr color space."""

    def __call__(self, img):
        """Convert image to YDbDr."""
        img = np.asarray(img, np.uint8)
        img = color.rgb2ydbdr(img)
        return img


class RGB2YPbPr(object):
    """Convert RGB PIL image to YPbPr color space."""

    def __call__(self, img):
        """Convert image to YPbPr."""
        img = np.asarray(img, np.uint8)
        img = color.rgb2ypbpr(img)
        return img


class RGB2YIQ(object):
    """Convert RGB PIL image to YIQ color space."""

    def __call__(self, img):
        """Convert image to YIQ."""
        img = np.asarray(img, np.uint8)
        img = color.rgb2yiq(img)
        return img


class RGB2CIERGB(object):
    """Convert RGB PIL image to CIE RGB color space."""

    def __call__(self, img):
        """Convert image to CIE RGB."""
        img = np.asarray(img, np.uint8)
        img = color.rgb2rgbcie(img)
        return img
