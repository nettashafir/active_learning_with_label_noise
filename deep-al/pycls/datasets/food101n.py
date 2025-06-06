from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm

import pycls.datasets.utils as ds_utils

class Food101N(Dataset):
    """
    Food101N dataset containing 310k images with ~20% label noise.
    The dataset uses Food-101 taxonomy but images are collected from different sources.
    """
    NUM_CLASSES = 101

    # (!) the a49e6fb26e356c6c41a37b5c18176355.jpg file is mentioned as filet_mignon in the imagelist.tsv, but as hot_and_sour_soup in the verified_train.tsv list (!)

    def __init__(self, root, train=True, transform=None, representation_model='simclr', only_features=False, validated_only=True, clean_test_set=True):
        self.root = root
        self.transform = transform
        self.train = train
        self.only_features = only_features
        self.validated_only = validated_only
        
        # Read class names
        with open(os.path.join(self.root, 'meta', 'classes.txt'), 'r') as f:
            lines = f.readlines()
            self.classes = [line.strip() for line in lines[1:]]  # Skip first line (header)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Load features using the provided representation model
        features = ds_utils.load_features("FOOD101N", train=train, representation_model=representation_model)
        normalized_features = features / np.linalg.norm(features, axis=1, keepdims=True)

        # Load image list and create mappings
        image_list = pd.read_csv(os.path.join(root, 'meta/imagelist.tsv'), sep='\t', header=0, names=['path'])

        # Load all images and their noisy labels
        self.data = []  # Will contain image paths
        self.targets = []  # Will contain noisy labels
        self.is_noisy = []


        # Set approximate noise rate based on paper
        self.verification_labels = {}

        # Load verification labels based on train/test split
        if clean_test_set:
            verify_file = 'verified_train_clean_test.tsv' if self.train else 'verified_val_clean_test.tsv'
        else:
            verify_file = 'verified_train.tsv' if self.train else 'verified_val.tsv'
        verify_data = pd.read_csv(os.path.join(root, "meta", verify_file), sep='\t', header=0, names=['path', 'verification'])


        # Build verification labels dictionary
        for idx, row in verify_data.iterrows():
            self.verification_labels[row['path']] = bool(int(row['verification']))

        # Initialize noise flags for verified samples and filter dataset if needed
        if self.validated_only or not self.train:
            # Keep track of indices of validated images
            for img_path, verified in self.verification_labels.items():
                class_name = img_path.split('/')[0]
                self.data.append(os.path.join(root, 'images', img_path))
                self.targets.append(self.class_to_idx[class_name])
                self.is_noisy.append(not verified)

            # Filter to keep only validated images
            self.is_noisy = np.array(self.is_noisy)
            self.noise_rate = np.mean(self.is_noisy.astype(int))
            self.noisy_labels = np.array(self.targets)

            # filter the relevant features
            indices_path = os.path.join(self.root, "meta", f"indices_{verify_file}.npy")
            print(f"Food101N | Looking for the indices of the validated images in {indices_path}")
            if not os.path.exists(indices_path):
                print("Food101N | No indices found. Load full dataset for finding indices of validated images")
                valid_indices = self._find_indices_of_verify_file(verify_file)
            else:
                print("Food101N | Loading indices of validated images")
                valid_indices = np.load(indices_path)
            self.features = features[valid_indices]
            self.normalized_features = normalized_features[valid_indices]

        else:  # train and not validated_only

            # load all images
            total_data, total_targets = [], []
            for idx, row in tqdm(image_list.iterrows(), "Food101N | Loading 310,000 images"):
                class_name = row['path'].split('/')[0]
                total_data.append(os.path.join(self.root, 'images', row['path']))
                total_targets.append(self.class_to_idx[class_name])

            # load the test set
            verify_test_file = 'verified_val_clean_test.tsv' if clean_test_set else 'verified_val.tsv'
            verify_test_data = pd.read_csv(os.path.join(self.root, "meta", verify_test_file), sep='\t', header=0, names=['path', 'verification'])

            # remove the test samples
            self.features = []
            self.normalized_features = []
            verify_test_data_set = {os.path.join(root, 'images', row['path']) for _, row in verify_test_data.iterrows()}
            for i, img_path in enumerate(total_data):
                if img_path not in verify_test_data_set:
                    self.data.append(img_path)  # total_data[i]
                    self.targets.append(total_targets[i])
                    self.features.append(features[i])
                    self.normalized_features.append(normalized_features[i])

            self.features = np.array(self.features)
            self.normalized_features = np.array(self.normalized_features)
            self.noise_rate = 0.20  # estimated
            self.is_noisy = np.full(len(self.data), False)
            self.noisy_labels = np.array(self.targets)

    def __getitem__(self, index):
        """
        Returns:
            tuple: (image, target, index) where target is index of the target class.
            If train=True, returns (image, noisy_label, true_label, index)
        """
        target = self.targets[index]
        if self.only_features:
            img = self.features[index]
        else:
            img_path = self.data[index]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)

        if self.train:
            # During training, return both noisy label and verification status if available
            noisy_label = target
            return img, noisy_label, target, index
        
        return img, target, index

    def __len__(self):
        return len(self.data)

    def _find_indices_of_verify_file(self, verify_file):
        # Load image list and create mappings
        image_list = pd.read_csv(os.path.join(self.root, 'meta/imagelist.tsv'), sep='\t', header=0, names=['path'])

        # Load all images and their noisy labels
        data = []  # Will contain image paths

        for idx, row in tqdm(image_list.iterrows(), "Food101N | Loading 310,000 images"):
            data.append(row['path'])

        # Keep track of indices of validated images
        valid_indices = []
        sub_data = []
        for i, img_path in enumerate(data):
            if img_path in self.verification_labels:
                valid_indices.append(i)
                sub_data.append(img_path)

        # save the indices

        np.save(os.path.join(self.root, "meta", f"indices_{verify_file}"), valid_indices)
        print(f"Food101N | Indices saved for future use in {self.root}/meta/indices_{verify_file}.npy")
        return valid_indices


if __name__ == "__main__":
    # Example usage and testing
    save_dir = '/path/to/food101n'
    
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])
    
    # Create dataset instances with all images
    train_dataset = Food101N(save_dir, train=True, transform=transform_train)
    test_dataset = Food101N(save_dir, train=False, transform=transform_test)
    print(f"Full training set size: {len(train_dataset)}")
    print(f"Full test set size: {len(test_dataset)}")
    
    # Create dataset instances with only validated images
    train_dataset_val = Food101N(save_dir, train=True, transform=transform_train, validated_only=True)
    test_dataset_val = Food101N(save_dir, train=False, transform=transform_test, validated_only=True)
    print(f"Validated training set size: {len(train_dataset_val)}")
    print(f"Validated test set size: {len(test_dataset_val)}")
    
    # Print some statistics about noise in validated sets
    print(f"Training set noise rate: {train_dataset_val.is_noisy.mean():.3f}")
    print(f"Test set noise rate: {test_dataset_val.is_noisy.mean():.3f}")
