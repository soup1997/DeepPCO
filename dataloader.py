#!/usr/bin/env python3

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset


import os
from natsort import natsorted
from PIL import Image
import numpy as np


kitti_time = {0: [0, 4540],
              1: [0, 1100],
              2: [0, 4660],
              4: [0, 270],
              5: [0, 2760],
              6: [0, 1100],
              7: [0, 1100],
              8: [1100, 5170],
              9: [0, 1590],
              10: [0, 1200]}


class KittiDataset(Dataset):
    def __init__(self, root_dir, sequence, valid_time):
        self.root_dir = root_dir
        self.sequence = sequence
        self.idx = valid_time

        # Set (img, gt) paths
        self.image_dir = os.path.join(root_dir, f'seq{sequence:02d}', 'img')
        self.pose_dir = os.path.join(root_dir, f'seq{sequence:02d}', f'pose_seq{sequence:02d}.txt')

        # Load (img, gt) files
        self.image_files = natsorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])[valid_time[0]: valid_time[1]+1]
        self.pose_file = self._load_poses()

        # According to sequence, apply different transformations
        self.transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Resize((64, 1024))])

    def _load_poses(self):
        pose_data = np.loadtxt(self.pose_dir)
        pose_data = torch.tensor(pose_data, dtype=torch.float32)
        return pose_data

    def _stack_image(self, img1, img2):
        stacked_img = torch.cat((img1, img2), dim=0)
        return stacked_img

    def __len__(self):  # Specifies the the size of dataset
        return len(self.image_files) - 1

    def __getitem__(self, idx):
        img1_name = self.image_files[idx]
        img2_name = self.image_files[idx+1]

        img1_path = os.path.join(self.image_dir, img1_name)
        img2_path = os.path.join(self.image_dir, img2_name)

        img1_tensor = self.transforms(np.array(Image.open(img1_path)))
        img2_tensor = self.transforms(np.array(Image.open(img2_path)))

        stacked_img = self._stack_image(img1_tensor, img2_tensor)
        ground_truth = self.pose_file[idx]

        return stacked_img, ground_truth


def calcultate_norm(dataset):
    mean_ = np.array([np.mean(x.numpy(), axis=(1, 2)) for x, _ in dataset])

    mean_r = mean_[:, 0].mean()
    mean_g = mean_[:, 1].mean()
    mean_b = mean_[:, 2].mean()

    std_ = np.array([np.std(x.numpy(), axis=(1, 2)) for x, _ in dataset])

    std_r = std_[:, 0].mean()
    std_g = std_[:, 1].mean()
    std_b = std_[:, 2].mean()

    return (mean_r, mean_g, mean_b), (std_r, std_g, std_b)


def load_dataset(root_dir, batch_size=64):
    train_datasets = []
    valid_datasets = []
    test_datasets = []

    for seq, valid_time in kitti_time.items():
        dataset = KittiDataset(
            root_dir=root_dir, sequence=seq, valid_time=valid_time)

        '''
        mean_, std_ = calcultate_norm(dataset)
        print(mean_, std_)
        '''

        if seq == 9 or seq == 10:
            test_datasets.append(dataset)

        else:
            train_datasets.append(dataset)

    train_loader = DataLoader(dataset=ConcatDataset(train_datasets),
                              batch_size=batch_size,
                              shuffle=True)

    # valid_loader = DataLoader(dataset=ConcatDataset(valid_datasets),
    #                           batch_size=batch_size,
    #                           shuffle=True)

    test_loader = DataLoader(dataset=ConcatDataset(test_datasets),
                             batch_size=batch_size,
                             shuffle=True)

    # return train_loader, valid_loader, test_loader
    return train_loader, test_loader


if __name__ == '__main__':
    root_dir = '/home/smeet/catkin_ws/src/PointFlow-Odometry/dataset/custom_sequence/'
    train_loader, valid_loader, test_loader = load_dataset(root_dir=root_dir, batch_size=64)

    print("Train Loader Length:", len(train_loader))
    print("Valid Loader Length:", len(valid_loader))
    print("Test Loader Length:", len(test_loader))