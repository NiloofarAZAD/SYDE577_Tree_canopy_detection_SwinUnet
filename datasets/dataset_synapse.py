import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from scipy import ndimage
from scipy.ndimage import zoom



def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=3, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label



class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size  

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        h, w, _ = image.shape

        
        if h != self.output_size or w != self.output_size:
            scale_h = self.output_size / h
            scale_w = self.output_size / w
            image = zoom(image, (scale_h, scale_w, 1), order=3)
            label = zoom(label, (scale_h, scale_w), order=0)

        
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        label = torch.from_numpy(label.astype(np.int64))

        return {"image": image, "label": label}



class Synapse_dataset(Dataset):
    """
    Compatible with original Synapse trainer.py
    but uses:
        base_dir/train_images/*.tif
        base_dir/train_masks/*.png
        list_dir/train.txt (filenames without extension)
    """

    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split

        list_file = os.path.join(list_dir, split + ".txt")
        with open(list_file, "r") as f:
            self.sample_list = [x.strip() for x in f.readlines()]

        self.image_dir = os.path.join(base_dir, "train_images")
        self.mask_dir = os.path.join(base_dir, "train_masks")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        name = self.sample_list[idx]

        img_path = os.path.join(self.image_dir, name + ".tif")
        mask_path = os.path.join(self.mask_dir, name + ".png")

        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0

        label = np.array(Image.open(mask_path), dtype=np.int64)

        sample = {"image": image, "label": label}

        if self.transform:
            sample = self.transform(sample)

        sample["case_name"] = name
        return sample
