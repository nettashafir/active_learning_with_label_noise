import os
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import torchvision.datasets as datasets
from tinyimagenet import TinyImageNet as TinyImageNetClass

from typing import Any


class TinyImageNet(TinyImageNetClass):
    """`Tiny ImageNet Classification Dataset.

    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        samples (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """
    def __init__(self, root: str, split: str = 'train', noise_type="clean_label", noise_rate=0.0, exp_dir=None, transform=None, test_transform=None, only_features=False, **kwargs: Any) -> None:
        super(TinyImageNet, self).__init__(Path(root), split=split)

        # self.root = root
        # assert self.check_root(), "Something is wrong with the Tiny ImageNet dataset path. Download the official dataset zip from http://cs231n.stanford.edu/tiny-imagenet-200.zip and unzip it inside {}.".format(self.root)
        # wnid_to_classes = self.load_wnid_to_classes()
        # self.wnids = self.classes
        # self.wnid_to_idx = self.class_to_idx
        # self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        # self.class_to_idx = {cls: idx
        #                      for idx, clss in enumerate(self.classes)
        #                      for cls in clss}

        # Tiny ImageNet val directory structure is not similar to that of train's
        # So a custom loading function is necessary
        self.split = datasets.utils.verify_str_arg(split, "split", ("train", "val", "test"))
        self.no_aug = False
        self.transform = transform
        self.test_transform = test_transform
        self.only_features = only_features
        if self.split == 'train':
            self.train = True
            self.features = np.load('/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/tiny-imagenet_simclr/pretext/features_seed1.npy')
        elif self.split == 'val':
            self.train = False
            self.features = np.load('/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/tiny-imagenet_simclr/pretext/test_features_seed1.npy')
            self.root = root
            # self.imgs, self.targets = self.load_val_data()
            # self.samples = [(self.imgs[idx], self.targets[idx]) for idx in range(len(self.imgs))]
            # self.root = os.path.join(self.root, 'val')
        elif self.split == 'test':
            self.train = False
            # self.features = np.load('/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/tiny-imagenet_simclr/pretext/test_features_seed1.npy')
        else:
            raise NotImplementedError(f"Split {self.split} not implemented for TinyImageNet")

        # Handle noisy labels
        # self.targets = np.array([s[1] for s in self.samples])

        if self.split == 'train':
            self.noise_type = noise_type
            save_noise = True

            exp_dir = exp_dir if exp_dir is not None else os.getcwd()
            noise_path = os.path.join(exp_dir, "noisy_labels.npy")
            use_saved_noise = exp_dir != "" and not exp_dir.endswith("test")
            if use_saved_noise and os.path.exists(noise_path):
                print(f"TinyImagenet | Loading noisy labels from existed directory")
                noisy_labels = np.load(noise_path)
                save_noise = False
            elif noise_type == "clean_label":
                noisy_labels = np.array(self.targets).copy()
            elif noise_type == "sym":
                noisy_labels = np.array(self.targets).copy()
                if noise_rate > 0:
                    print(f"TinyImagenet | Generating symmetric noise with rate {noise_rate}")
                    noisy_labels = np.array(self.targets).copy()
                    all_labels = np.arange(200)
                    noise_idx = np.random.choice(len(self.targets), int(noise_rate * len(self.targets)), replace=False)
                    valid_labels = all_labels != noisy_labels[noise_idx, np.newaxis]
                    new_labels = np.apply_along_axis(lambda x: np.random.choice(all_labels[x]), 1, valid_labels)
                    noisy_labels[noise_idx] = new_labels
            else:
                raise NotImplementedError(f"Noise type {noise_type} not implemented for TinyImageNet")

            self.noisy_labels = noisy_labels
            self.is_noisy = np.asarray(self.noisy_labels) != np.asarray(self.targets)
            self.noise_rate = np.mean(self.is_noisy.astype(int))
            if save_noise:
                np.save(noise_path, self.noisy_labels)

    # Split folder is used for the 'super' call. Since val directory is not structured like the train, 
    # we simply use train's structure to get all classes and other stuff
    # @property
    # def split_folder(self) -> str:
    #     return os.path.join(self.root, 'train')

    # def load_val_data(self):
    #     imgs, targets = [], []
    #     with open(os.path.join(self.root, 'val', 'val_annotations.txt'), 'r') as file:
    #         for line in file:
    #             if line.split()[1] in self.wnids:
    #                 img_file, wnid = line.split('\t')[:2]
    #                 imgs.append(os.path.join(self.root, 'val', wnid, 'images', img_file))
    #                 targets.append(wnid)
    #     targets = np.array([self.wnid_to_idx[wnid] for wnid in targets])
    #     return imgs, targets

    # def load_wnid_to_classes(self):
    #     wnid_to_classes = {}
    #     with open(os.path.join(self.root, 'words.txt'), 'r') as file:
    #         lines = file.readlines()
    #         lines = [x.split('\t') for x in lines]
    #         wnid_to_classes = {x[0]:x[1].strip() for x in lines}
    #     return wnid_to_classes

    # def check_root(self):
    #     tinyim_set = ['words.txt', 'wnids.txt', 'train', 'val', 'test']
    #     existed = [False] * len(tinyim_set)
    #     # for x in os.scandir(self.root):
    #     #     if x.name not in tinyim_set:
    #     #         return False
    #     for x in os.scandir(f"{self.root}/tiny-imagenet-200"):
    #         if x.name in tinyim_set:
    #             idx = tinyim_set.index(x.name)
    #             existed[idx] = True
    #
    #     if not all(existed):
    #         return False
    #     return True

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        img = self.loader(path)

        if self.only_features:
            img = self.features[index]
        else:
            if self.no_aug:
                if self.test_transform is not None:
                    img = self.test_transform(img)
            else:
                if self.transform is not None:
                    img = self.transform(img)

        if self.train:
            noisy_label = self.noisy_labels[index]
            return img, noisy_label, target, index
        return img, target, index