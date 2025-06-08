"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os

import numpy as np
import torch
import torchvision.datasets as datasets
import torch.utils.data as data
from PIL import Image
from torchvision import transforms as tf
from glob import glob
from typing import Any
from tqdm import tqdm

import pycls.datasets.utils as ds_utils


class ImageNet(datasets.ImageFolder):
    def __init__(self, root: str, split: str = 'train', transform=None, test_transform=None, only_features=False,
                 **kwargs: Any) -> None:
        """
        @param root: Parent directory of classes directories. Each class has another dir.
        @param split: val, train or test
        @param transform:
        """
        self.root = root
        self.test_transform = test_transform
        self.no_aug = False

        assert self.check_root(), "Something is wrong with the ImageNetCLS dataset path"
        self.split = datasets.utils.verify_str_arg(split, "split", ("train", "val"))
        wnid_to_classes = self.load_wnid_to_classes()

        super(ImageNet, self).__init__(root=os.path.join(root, 'dist', split), **kwargs)
        self.root = root

        self.transform = transform
        # self.resize = tf.Resize(256)
        self.only_features = only_features

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}  # classes names -> idx

        if self.split == 'train':
            self.imgs, self.targets = self.load_train_data()
            self.samples = [(self.imgs[idx], self.targets[idx]) for idx in range(len(self.imgs))]
            self.root = os.path.join(self.root, 'dist', 'train')

        elif self.split == 'val':
            self.imgs, self.targets, self.targets_wnid = self.load_val_data()
            self.samples = [(self.imgs[idx], self.targets[idx]) for idx in range(len(self.imgs))]
            self.root = os.path.join(self.root, 'dist', 'val')

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.samples[index]
        if self.split == "train":
            folder = path.split('_')[0]
            sample = self.loader(os.path.join(self.root, folder, path))
        else:
            wnid = self.targets_wnid[index]
            sample = self.loader(os.path.join(self.root, wnid, path))

        if self.only_features:
            sample = self.features[index]
        else:
            if self.no_aug:
                if self.test_transform is not None:
                    sample = self.test_transform(sample)
            else:
                if self.transform is not None:
                    sample = self.transform(sample)

        return sample, target

    def check_root(self):
        check_list = ['labels.txt', 'dist', 'class-ids-TRAIN.npy', 'class-ids-VAL.npy', 'class-names-TRAIN.npy',
                      'class-names-VAL.npy', 'entries-TEST.npy', 'entries-VAL.npy', 'entries-TRAIN.npy']
        dirs = [x.name for x in os.scandir(self.root)]
        for x in check_list:
            if x not in dirs:
                return False

        check_list = ['val', 'train', 'test']
        for x in os.scandir(os.path.join(self.root, 'dist')):
            if x.name not in check_list:
                return False
        return True

    def load_train_data(self):
        extra_data = np.load(os.path.join(self.root, 'entries-TRAIN.npy'), allow_pickle=True)
        targets_wnid = [item[2] for item in extra_data if item[2] in self.wnids]
        imgs = [item[2] + '_' + str(item[0]) + '.JPEG' for item in extra_data if item[2] in self.wnids]
        targets = [self.wnid_to_idx[wnid] for wnid in targets_wnid]
        return imgs, targets

    def load_val_data(self):
        extra_data = np.load(os.path.join(self.root, 'entries-VAL.npy'), allow_pickle=True)
        targets_wnid = [item[2] for item in extra_data if item[2] in self.wnids]
        imgs = ['ILSVRC2012_val_' + '{:0>8}'.format(int(item[0])) + '.JPEG' for item in extra_data if
                item[2] in self.wnids]
        targets = np.array([self.wnid_to_idx[wnid] for wnid in targets_wnid])
        return imgs, targets, targets_wnid

    def load_wnid_to_classes(self):
        with open(os.path.join(self.root, 'labels.txt'), 'r') as file:
            wnid_to_classes = eval(file.read())
        return wnid_to_classes


class ImageNetSubset(data.Dataset):
    def __init__(self, cfg, subset_file, root: str, exp_dir: str = ".", split: str = 'train', transform=None, test_transform=None,
                 only_features=False, representation_model="dinov2", noise_type="clean_label", noise_rate=0.0, no_indices=False) -> None:
        """
        @param root: Parent directory of classes directories. Each class has another dir.
        @param split: val, train or test
        @param transform:
        """
        super(ImageNetSubset, self).__init__()
        self.root = root
        assert self.check_root(), "Something is wrong with the ImageNetCLS dataset path"

        self.test_transform = test_transform
        self.no_aug = False
        self.transform = transform
        self.split = datasets.utils.verify_str_arg(split, "split", ("train", "val"))
        self.train = (split == 'train')
        self.only_features = only_features
        self.no_indices = no_indices

        # Read the subset of classes to include (sorted)
        with open(subset_file, 'r') as f:
            result = f.read().splitlines()
        subdirs, class_names = [], []
        for line in result:
            subdir, class_name = line.split(' ', 1)
            subdirs.append(subdir)
            class_names.append(class_name)

        # Gather the files (sorted)
        samples = []
        targets = []
        imgs = []
        for i, subdir in enumerate(subdirs):
            files = sorted(glob(os.path.join(self.root, 'dist', self.split, subdir, '*.JPEG')))
            for f in files:
                samples.append((f, i))
                targets.append(i)
                imgs.append(f)
        self.targets = targets
        self.imgs = imgs
        self.samples = samples  # (file, class_idx)
        self.classes = class_names  # idx_to_class[class_idx] = 'class_name'
        self.dataset_name = f"ImageNet{len(self.classes)}"
        self.class_to_idx = {cls: idx for idx, clss in enumerate(self.classes) for cls in clss}  # classes names -> idx
        self.wnids = subdirs  # subdirs[class_idx] = 'class_winds'

        # Load features
        self.features = self.load_features_for_subset(representation_model)
        self.normalized_features = self.features / np.linalg.norm(self.features, axis=1, keepdims=True)
        self.root = os.path.join(self.root, 'dist', self.split)

        # load noise
        if self.train:
            self.noise_type = noise_type
            save_noise = True

            noise_path = f"../../../data/imagenet{len(class_names)}_noise_type_{noise_type}_noise_rate_{noise_rate}_noisy_labels_seed_{cfg.RNG_SEED}.npy"
            use_saved_noise = exp_dir != "" and not exp_dir.endswith("test")
            if use_saved_noise and os.path.exists(noise_path) and noise_type not in ['clean_label', 'noisy_label']:
                print(f"{self.dataset_name} | Loading noisy labels from existed directory")
                noisy_labels = np.load(noise_path)
                save_noise = False

            elif self.noise_type in ['clean_label']:
                print(f"{self.dataset_name} | No noise added")
                noisy_labels = np.array(self.targets).copy()

            elif self.noise_type in ["sym", "asym"]:
                noisy_labels = np.array(self.targets).copy()
                if noise_rate > 0.0:
                    noise_idx = np.random.choice(len(self.targets), int(noise_rate * len(self.targets)), replace=False)

                    if self.noise_type == "sym":
                        print(f"{self.dataset_name} | Generating symmetric noise with rate {noise_rate}")
                        all_labels = np.arange(len(self.classes))
                        valid_labels = all_labels != noisy_labels[noise_idx, np.newaxis]
                        new_labels = np.apply_along_axis(lambda x: np.random.choice(all_labels[x]), 1, valid_labels)
                        noisy_labels[noise_idx] = new_labels

                    if self.noise_type == "asym":
                        print(f"{self.dataset_name} | Generating asymmetric noise with rate {noise_rate}")
                        confusion_matrix = np.load("../../../data/confusion_matrix_imagenet50_epochs_10.npy")
                        noisy_labels = ds_utils.add_asym_label_noise(noisy_labels, confusion_matrix, noise_rate)

            else:
                raise ValueError(f"Unknown noise type: {self.noise_type} with dataset IMAGENET")

            self.noisy_labels = noisy_labels
            self.is_noisy = np.asarray(self.noisy_labels) != np.asarray(self.targets)
            self.noise_rate = np.mean(self.is_noisy.astype(int))
            if save_noise:
                np.save(noise_path, self.noisy_labels)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        # sample = self.resize(sample)

        if self.only_features:
            sample = self.features[index]
        else:
            if self.no_aug:
                if self.test_transform is not None:
                    sample = self.test_transform(sample)
            else:
                if self.transform is not None:
                    sample = self.transform(sample)

        if self.no_indices:
            return sample, target
        if self.train:
            noisy_label = self.noisy_labels[index]
            return sample, noisy_label, target, index
        else:
            return sample, target, index

    def check_root(self):
        check_list = ['labels.txt', 'dist', 'class-ids-TRAIN.npy', 'class-ids-VAL.npy', 'class-names-TRAIN.npy',
                      'class-names-VAL.npy', 'entries-TEST.npy', 'entries-VAL.npy', 'entries-TRAIN.npy']
        dirs = [x.name for x in os.scandir(self.root)]
        for x in check_list:
            if x not in dirs:
                return False

        check_list = ['val', 'train', 'test']
        for x in os.scandir(os.path.join(self.root, 'dist')):
            if x.name not in check_list:
                return False
        return True

    def _load_features_map(self, full_dataset_features):
        """
        create mapping of image path to its vector
        @return:
        """
        extra_data = np.load(os.path.join(self.root, f'entries-{self.split.upper()}.npy'), allow_pickle=True)  # all images

        result = {}
        for image_details, features_vector in tqdm(zip(extra_data, full_dataset_features), desc="Creating features map"):
            is_in_subset = image_details[2] in self.wnids
            if is_in_subset:
                if self.split == 'train':
                    image = image_details[2] + '_' + str(image_details[0]) + '.JPEG'
                    image_path = os.path.join(self.root, 'dist', 'train', image_details[2], image)
                else:
                    image = 'ILSVRC2012_val_' + '{:0>8}'.format(int(image_details[0])) + '.JPEG'
                    image_path = os.path.join(self.root, 'dist', 'val', image_details[2], image)

                result[image_path] = features_vector

        return result

    def load_features_for_subset(self, representation_model):
        try:
            features = ds_utils.load_features(self.dataset_name.upper(), train=self.train, representation_model=representation_model)
            return features

        except KeyError:
            print(f"{self.dataset_name} | Features not exists. Extracting and saving features")
            full_dataset_features = ds_utils.load_features("IMAGENET", train=self.train, representation_model=representation_model)
            features_map = self._load_features_map(full_dataset_features)
            features = [features_map[img][None, :] for img in self.imgs]
            if isinstance(features[0], np.ndarray):
                features = np.concatenate(features)
            else:
                features = torch.cat(features, dim=0)

            # save the features
            out_dir = f"../../../data/representations_bank/{self.dataset_name.lower()}_{representation_model}"
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            ds_utils.set_new_features(self.train, self.dataset_name.upper(), representation_model, features, out_dir)

            return features

