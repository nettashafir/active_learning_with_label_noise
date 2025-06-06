import os
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import pycls.datasets.utils as ds_utils

class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, noise_root, noise_type, noise_rate, exp_dir, train, transform, test_transform,
                 download=True, only_features=False, project_features=False, representation_model="simclr"):
        super(CIFAR10, self).__init__(root, train, transform=transform, download=download)
        self.test_transform = test_transform
        self.no_aug = False
        self.only_features = only_features
        self.features = ds_utils.load_features("CIFAR10", train=train, representation_model=representation_model, project=project_features)
        self.normalized_features = self.features / np.linalg.norm(self.features, axis=1, keepdims=True)

        # load noise
        if self.train:
            self.noise_type = noise_type
            self.noise_rate = noise_rate
            save_noise = True
            self.transition = {0: 0, 1: 1, 2: 0, 3: 5, 4: 7, 5: 3, 6: 6, 7: 7, 8: 8, 9: 1}  # class transition for asymmetric noise

            noise_path = os.path.join(exp_dir, "noisy_labels.npy")
            use_saved_noise = exp_dir != "" and not exp_dir.endswith("test")
            if use_saved_noise and os.path.exists(noise_path):
                print(f"CIFAR10 | Loading noisy labels from existed directory")
                noisy_labels = np.load(noise_path)
                save_noise = False

            elif self.noise_type in ["clean_label", "worse_label", "aggre_label", "random_label1", "random_label2", "random_label3"]:
                print(f"CIFAR10 | Loading CIFAR10N human noise from type {self.noise_type}")
                noise_file = torch.load(f'{noise_root}/CIFAR-10_human.pt')
                noisy_labels = noise_file[self.noise_type]
                # self.noise_type in ['clean_label', 'worse_label', 'aggre_label', 'random_label1', 'random_label2', 'random_label3']

            elif self.noise_type == 'sym' or self.noise_type == 'asym':
                noisy_labels = np.array(self.targets).copy()
                noise_idx = np.random.choice(len(self.targets), int(self.noise_rate * len(self.targets)), replace=False)

                if self.noise_type == 'sym':
                    print(f"CIFAR10 | Generating symmetric noise with rate {self.noise_rate}")
                    # Symmetric noise: Randomly assign labels to noise_idx
                    all_labels = np.arange(10)
                    if self.noise_rate > 0.0:
                        valid_labels = all_labels != noisy_labels[noise_idx, np.newaxis]
                        new_labels = np.apply_along_axis(lambda x: np.random.choice(all_labels[x]), 1, valid_labels)
                        noisy_labels[noise_idx] = new_labels

                    # Equivalent to:
                    # for idx in noise_idx:
                    #     # Generate a new label that's different from the original
                    #     new_label = np.random.choice([label for label in range(10) if label != self.targets[idx]])
                    #     noisy_label[idx] = new_label

                elif self.noise_type == 'asym':
                    print(f"CIFAR10 | Generating asymmetric noise with rate {self.noise_rate}")
                    # Asymmetric noise: Use transition matrix
                    noisy_labels[noise_idx] = np.array([self.transition[label] for label in np.asarray(self.targets)[noise_idx]])

            else:
                raise ValueError(f"Unknown noise type: {self.noise_type} with dataset CIFAR10")

            self.noisy_labels = noisy_labels
            self.is_noisy = np.asarray(self.noisy_labels) != np.asarray(self.targets)
            self.noise_rate = np.mean(self.is_noisy.astype(int))
            if save_noise:
                np.save(noise_path, self.noisy_labels)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, noisy_label, true_label) where true_label is index of the true_label class.
        """
        img, true_label = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
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
            return img, noisy_label, true_label, index
        else:
            return img, true_label, index


class CIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, cfg, root, noise_root, noise_type, noise_rate, exp_dir, train, transform, test_transform,
                 download=True, only_features=False, project_features=False, representation_model="simclr",
                 ):
        super(CIFAR100, self).__init__(root, train, transform=transform, download=download)
        self.test_transform = test_transform
        self.no_aug = False
        self.only_features = only_features
        self.features = ds_utils.load_features("CIFAR100", train=train, representation_model=representation_model, project=project_features)
        self.normalized_features = self.features / np.linalg.norm(self.features, axis=1, keepdims=True)
        self.train = train

        # load noise
        if self.train:
            self.noise_type = noise_type
            save_noise = True

            noise_path = os.path.join(root, f"cifar100_noise_type_{noise_type}_noise_rate_{noise_rate}_noisy_labels_seed_{cfg.RNG_SEED}.npy")
            use_saved_noise = exp_dir != "" and not exp_dir.endswith("test")
            if use_saved_noise and os.path.exists(noise_path) and noise_type not in ['clean_label', 'noisy_label']:
                print(f"CIFAR100 | Loading noisy labels from existed directory")
                noisy_labels = np.load(noise_path)
                save_noise = False

            elif self.noise_type in ['clean_label', 'noisy_label']:
                print(f"CIFAR100 | Loading CIFAR100N human noise from type {self.noise_type}")
                noise_file = torch.load(f'{noise_root}/CIFAR-100_human.pt')
                noisy_labels = noise_file[self.noise_type]

            elif self.noise_type in ['sym', "asym"]:
                noisy_labels = np.array(self.targets).copy()
                if noise_rate > 0.0:
                    noise_idx = np.random.choice(len(self.targets), int(noise_rate * len(self.targets)), replace=False)

                    if self.noise_type == "sym":
                        print(f"CIFAR100 | Generating symmetric noise with rate {noise_rate}")
                        all_labels = np.arange(100)
                        valid_labels = all_labels != noisy_labels[noise_idx, np.newaxis]
                        new_labels = np.apply_along_axis(lambda x: np.random.choice(all_labels[x]), 1, valid_labels)
                        noisy_labels[noise_idx] = new_labels

                    else:  # asymmetric noise
                        print(f"CIFAR100 | Generating asymmetric noise with rate {noise_rate}")
                        confusion_matrix = np.load("/cs/labs/daphna/nettashaf/TypiClustNoisy/experiments/45_create_confusion_matrix/confusion_matrix_cifar100_epochs_10.npy")
                        noisy_labels = ds_utils.add_asym_label_noise(noisy_labels, confusion_matrix, noise_rate)

                        # Another Option: Use deterministic transition matrix
                        # transition = {0: 57, 1: 91, 2: 11, 3: 42, 4: 4, 5: 5, 6: 18, 7: 14, 8: 48, 9: 61, 10: 28, 11: 2, 12: 68, 13: 90, 14: 7, 15: 31, 16: 16, 17: 37, 18: 6, 19: 21, 20: 84, 21: 19, 22: 22, 23: 71, 24: 24, 25: 94, 26: 79, 27: 27, 28: 10, 29: 78, 30: 55, 31: 15, 32: 67, 33: 60, 34: 63, 35: 35, 36: 74, 37: 17, 38: 38, 39: 87, 40: 86, 41: 81, 42: 3, 43: 88, 44: 93, 45: 45, 46: 98, 47: 52, 48: 8, 49: 49, 50: 80, 51: 53, 52: 47, 53: 51, 54: 70, 55: 30, 56: 56, 57: 0, 58: 58, 59: 96, 60: 33, 61: 9, 62: 92, 63: 34, 64: 75, 65: 65, 66: 66, 67: 32, 68: 12, 69: 89, 70: 54, 71: 23, 72: 95, 73: 73, 74: 36, 75: 64, 76: 76, 77: 99, 78: 29, 79: 26, 80: 50, 81: 41, 82: 82, 83: 83, 84: 20, 85: 85, 86: 40, 87: 39, 88: 43, 89: 69, 90: 13, 91: 1, 92: 62, 93: 44, 94: 25, 95: 72, 96: 59, 97: 97, 98: 46, 99: 77}
                        # noisy_labels[noise_idx] = np.array([transition[label] for label in np.asarray(self.targets)[noise_idx]])

                # Equivalent to:
                # for idx in noise_idx:
                #     # Generate a new label that's different from the original
                #     new_label = np.random.choice([label for label in range(100) if label != self.targets[idx]])
                #     noisy_label[idx] = new_label

            else:
                raise ValueError(f"Unknown noise type: {self.noise_type} with dataset CIFAR100")

            self.noisy_labels = noisy_labels
            self.is_noisy = np.asarray(self.noisy_labels) != np.asarray(self.targets)
            self.noise_rate = np.mean(self.is_noisy.astype(int))
            if save_noise:
                np.save(noise_path, self.noisy_labels)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, noisy_label, true_label) where true_label is index of the true_label class.
        """
        img, true_label = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
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
            return img, noisy_label, true_label, index
        else:
            return img, true_label, index


class STL10(torchvision.datasets.STL10):
    def __init__(self, root, train, transform, test_transform, download=True):
        super(STL10, self).__init__(root, train, transform=transform, download=download)
        self.test_transform = test_transform
        self.no_aug = False
        self.targets = self.labels

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, noisy_target, target) where target is index of the target class.
        """
        img, target, noisy_target = self.data[index], self.targets[index], self.noisy_targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.only_features:
            img = self.features[index]
        else:
            if self.no_aug:
                if self.test_transform is not None:
                    img = self.test_transform(img)
            else:
                if self.transform is not None:
                    img = self.transform(img)

        return img, noisy_target, target


class MNIST(torchvision.datasets.MNIST):
    def __init__(self, root, train, transform, test_transform, download=True):
        super(MNIST, self).__init__(root, train, transform=transform, download=download)
        self.test_transform = test_transform
        self.no_aug = False

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        
        if self.no_aug:
            if self.test_transform is not None:
                img = self.test_transform(img)            
        else:
            if self.transform is not None:
                img = self.transform(img)


        return img, target


class SVHN(torchvision.datasets.SVHN):
    def __init__(self, root, train, transform, test_transform, download=True):
        super(SVHN, self).__init__(root, train, transform=transform, download=download)
        self.test_transform = test_transform
        self.no_aug = False

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        
        if self.no_aug:
            if self.test_transform is not None:
                img = self.test_transform(img)            
        else:
            if self.transform is not None:
                img = self.transform(img)


        return img, target

#
# class Clothing1M(torch.utils.data.Dataset):
#     def __init__(self, root, train, valid, test):
#         transform_train = transforms.Compose([transforms.Resize((256, 256)),
#                                               transforms.RandomCrop(224),
#                                               transforms.RandomHorizontalFlip(),
#                                               transforms.ToTensor(),
#                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                                    std=[0.229, 0.224, 0.225])])
#         transform_test = transforms.Compose([transforms.Resize((224, 224)),
#                                              transforms.ToTensor(),
#                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                                   std=[0.229, 0.224, 0.225])])
#
#         self.root = root
#         if train==True:
#             flist = os.path.join(root, "annotations/noisy_train.txt")
#             self.transform = transform_train
#         if valid==True:
#             flist = os.path.join(root, "annotations/clean_val.txt")
#             self.transform = transform_test
#         if test==True:
#             flist = os.path.join(root, "annotations/clean_test.txt")
#             self.transform = transform_test
#         else:
#             raise ValueError("Invalid dataset type")
#
#         self.imlist = self._flist_reader(flist)
#         self.train = train
#
#     def __getitem__(self, index):
#         impath, target = self.imlist[index]
#         img = Image.open(impath).convert("RGB")
#         if self.transform is not None:
#             img = self.transform(img)
#         return index, img, target
#
#     def __len__(self):
#         return len(self.imlist)
#
#     def _flist_reader(self, flist):
#         imlist = []
#         with open(flist, 'r') as rf:
#             for line in rf.readlines():
#                 row = line.split(" ")
#                 impath = self.root + row[0]
#                 imlabel = row[1]
#                 imlist.append((impath, int(imlabel)))
#         return imlist