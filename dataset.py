from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import numpy as np


class supersound_dataset(Dataset):
    def __init__(self, data_path_list, data_transforms=None, label_transforms=None):
        self.data_path_list = data_path_list
        self.data_transforms = data_transforms
        self.label_transforms = label_transforms

    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, item):

        img_path = self.data_path_list[item]
        label_path = self.data_path_list[item].replace('image', 'label')
        img = Image.open(img_path).convert('L')
        try:
            point_label = Image.open(img_path.replace('image', 'train_point_r_mask'))
            point_label = np.array(point_label)
            point_weight = np.load(img_path.replace('image', 'train_point_r_weight').replace('bmp', 'npy'))
            bound_label = Image.open(img_path.replace('image', 'train_point_mask'))
            bound_label = np.array(bound_label)
            is_train = True
        except:
            is_train = False

        label = Image.open(label_path).convert('L')
        label = np.array(label)
        label[np.where(label == 15)] = 0
        if self.data_transforms is not None:
            img = self.data_transforms(img)
        if self.label_transforms is not None:
            label = self.label_transforms(label)
        if is_train:
            return img, torch.from_numpy(point_label), torch.from_numpy(bound_label), torch.from_numpy(label), torch.from_numpy(point_weight), img_path
        else:
            return img, torch.from_numpy(label), img_path












