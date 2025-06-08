from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import torch
import os

import pycls.datasets.utils as ds_utils


class WebVision(Dataset):
    """
    WebVision dataset loader
    """
    NUM_CLASSES = 1000  # Full WebVision has 1000 classes

    def __init__(self, root, train=True, transform=None, representation_model='dinov2_small', only_features=False,
                 num_class=50, image_source="flickr"):
        self.root = root
        self.transform = transform
        self.train = train
        self.only_features = only_features
        self.num_class = num_class  # Support for Mini-WebVision (50 classes)
        self.features = ds_utils.load_features(f"MINIWEBVISION_{image_source.upper()}", train=train, representation_model=representation_model)
        self.normalized_features = self.features / np.linalg.norm(self.features, axis=1, keepdims=True)

        # Load class names from synsets.txt
        with open(os.path.join(self.root, 'info/synsets.txt')) as f:
            lines = f.readlines()
        self.classes = []
        for line in lines[:num_class]:
            # Get first word after space and before comma, replace spaces with underscores
            class_name = line.split(' ', 1)[1].split(',')[0].strip().replace(' ', '_').lower()
            self.classes.append(class_name)

        if self.train:
            # Load training data
            with open(os.path.join(self.root, f'info/train_filelist_{image_source}.txt')) as f:
                lines = f.readlines()

            self.data = []
            self.noisy_labels = []
            self.targets = []  # Will be same as noisy_labels for consistency
            # indices = []

            for i, line in enumerate(lines):
                img_path, target = line.split()
                target = int(target)
                if target < self.num_class:
                    full_path = os.path.join(self.root, img_path)
                    self.data.append(full_path)
                    self.noisy_labels.append(target)
                    self.targets.append(target)
                    # indices.append(i)

            # Initialize noise-related attributes
            self.noise_rate = 0.2  # Estimated noise rate for WebVision
            self.is_noisy = np.full(len(self.data), False)

        else:
            # Load validation/test data
            with open(os.path.join(self.root, 'info/val_filelist.txt')) as f:
                lines = f.readlines()

            self.data = []
            self.targets = []
            # indices = []

            for i, line in enumerate(lines):
                img_path, target = line.split()
                target = int(target)
                if target < self.num_class:
                    full_path = os.path.join(self.root, 'val_images_resized', img_path)
                    self.data.append(full_path)
                    self.targets.append(target)
                    # indices.append(i)

        # self.features = total_features[indices]
        # self.normalized_features = total_normalized_features[indices]

    def __getitem__(self, index):
        target = self.targets[index]

        if self.only_features:
            img = self.features[index]
        else:
            img_path = self.data[index]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)

        if self.train:
            noisy_label = self.noisy_labels[index]  # same as target in this case
            return img, noisy_label, target, index
        return img, target, index

    def __len__(self):
        return len(self.targets)