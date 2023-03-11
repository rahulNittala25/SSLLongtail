import os
import sys
import pickle
import numpy as np
from PIL import Image
import torch.utils.data as data

import torchvision
import torch
from torchvision.transforms import transforms
from RandAugment import RandAugment
from RandAugment.augmentations import CutoutDefault
import json


class TransformTwice:
    def __init__(self, transform, transform2):
        self.transform = transform
        self.transform2 = transform2

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform2(inp)
        return out1, out2


def get_mscoco_glt(anno_path, phase, img_size=112, return_strong_labeled_set=False):

    assert img_size == 64 or img_size == 112, 'img size should only be 32 or 64!!!'
    assert phase in ["test_bl","test_bbl"]

    # compute dataset mean and std
    dataset_mean = (0.473, 0.429, 0.370)  # np.mean(base_dataset.data, axis=(0, 1, 2)) / 255
    print(dataset_mean)

    dataset_std = (0.277, 0.268, 0.274)  # np.std(base_dataset.data, axis=(0, 1, 2)) / 255
    print(dataset_std)

    # construct data augmentation
    # Augmentations.
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ])

    transform_strong = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ])
    transform_strong.transforms.insert(0, RandAugment(3, 4))
    transform_strong.transforms.append(CutoutDefault(int(img_size / 2)))

    transform_val = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ])

    train_labeled_dataset = MSCOCO_GLT(anno_path, "train_l", img_size, transform=transform_train)
    train_unlabeled_dataset = MSCOCO_GLT(anno_path, "train_ul", img_size, transform=TransformTwice(transform_train, transform_strong))
    test_dataset = MSCOCO_GLT(anno_path, phase, img_size, transform=transform_val)

    if return_strong_labeled_set:
        train_strong_labeled_dataset = MSCOCO_GLT(anno_path, "train_l", img_size, transform=transform_strong)
        return train_labeled_dataset, train_unlabeled_dataset, test_dataset, train_strong_labeled_dataset
    else:
        return train_labeled_dataset, train_unlabeled_dataset, test_dataset


class MSCOCO_GLT(data.Dataset):
    def __init__(self, anno_path, data_split="train_l", imgsize=112, transform=None, target_transform=None):
        assert imgsize == 64 or imgsize == 112, 'imgsize should only be 32 or 64'
        assert data_split in ['train_l', 'train_ul', 'val', 'test_bl', 'test_bbl'], "Illegal phase. Should be in - ['train_l', 'train_ul', 'val', 'test_bl', 'test_bbl']"
        self.imgsize = imgsize
        self.data_split = data_split
        self.transform = transform
        self.target_transform = target_transform
        
        with open(anno_path, 'r') as fp:
            self.annotations = json.load(fp)
        self.data = self.annotations[data_split]
        # Load Folder name(id) to class mappings and vice versa
        self.id2cat, self.cat2id = self.annotations['id2cat'], self.annotations['cat2id']
        # Load image paths and their corresponding labels, attributes and frequency categories(many, medium, few)
        self.img_paths, self.labels, self.attributes, self.frequencies = self.load_img_info()
        
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, label, rarity, attribute, index) if it is any of the test sets
            tuple: (image, label, rarity, index) if it is labelled train set
            tuple: (image, rarity, index) if it is unlabelled train set
        """
        path = self.img_paths[index]
        label = self.labels[index]
        rarity = self.frequencies[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)
            
        if self.target_transform is not None:
            target = self.target_transform(label)

        # intra-class attribute SHOULD NOT be used during training
        if ("train_l" not in self.data_split) and ("train_ul" not in self.data_split):
            attribute = self.attributes[index]
            return sample, label, rarity, attribute, index
        else:
            return sample, label, rarity, index

    def __len__(self):
        return len(self.labels)
    
    def load_img_info(self):
        img_paths = []
        labels = []
        attributes = []
        frequencies = []


        for path, label in self.data['label'].items():
            img_paths.append(self.data['path'][path])
            labels.append(int(label))
            frequencies.append(int(self.data['frequency'][path]))

            # intra-class attribute SHOULD NOT be used in training
            if self.data_split not in ['train_l', 'train_ul']:
                att_label = int(self.data['attribute'][path])
                attributes.append(att_label)

        return img_paths, labels, attributes, frequencies
