from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch

import pycls.datasets.utils as ds_utils


class Clothing1M(Dataset):
    """
    This dataset has approximately 38% label noise.
    """
    NUM_CLASSES = 14

    def __init__(self, root, train, transform, representation_model='dinov2_small', only_features=False):
        self.root = root
        self.transform = transform
        self.train = train
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}
        self.only_features = only_features
        self.classes = ['T-Shirt', 'Shirt', 'Knitwear', 'Chiffon', 'Sweater', 'Hoodie', 'Windbreaker', 'Jacket', 'Downcoat', 'Suit', 'Shawl', 'Dress', 'Vest', 'Underwear']
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.features = ds_utils.load_features("CLOTHING1M", train=train, representation_model=representation_model)
        self.normalized_features = self.features / np.linalg.norm(self.features, axis=1, keepdims=True)
        with open('%s/noisy_label_kv.txt'%self.root,'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = '%s/'%self.root+entry[0]
                self.train_labels[img_path] = int(entry[1])
        with open('%s/clean_label_kv.txt'%self.root,'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = '%s/'%self.root+entry[0]
                self.test_labels[img_path] = int(entry[1])

        if self.train:
            train_imgs = []
            with open('%s/noisy_train_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root+l
                    train_imgs.append(img_path)
            class_num = torch.zeros(self.NUM_CLASSES)
            self.data = []
            self.noisy_labels = []
            for i, impath in enumerate(train_imgs):
                self.data.append(impath)
                noisy_label = self.train_labels[impath]
                self.noisy_labels.append(noisy_label)
                class_num[noisy_label] += 1
            self.targets = self.noisy_labels.copy()

            # Clean train set
            clean_train_imgs = []
            with open('%s/clean_train_key_list.txt' % self.root, 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/' % self.root + l
                    clean_train_imgs.append(img_path)
            self.clean_data = []
            self.clean_targets = []
            for impath in clean_train_imgs:
                self.clean_data.append(impath)
                true_label = self.test_labels[impath]
                self.clean_targets.append(true_label)

            self.noise_rate = 0.38
            self.is_noisy = np.full(len(self.data), False)

        else:
            test_imgs = []
            with open('%s/clean_test_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root+l
                    test_imgs.append(img_path)
                    # self.test_targets.append(label)
            self.test_imgs = []
            self.test_targets = []
            for impath in test_imgs:
                self.test_imgs.append(impath)
                label = self.test_labels[impath]
                self.test_targets.append(label)
            self.data = self.test_imgs
            self.targets = self.test_targets

    def __getitem__(self, index):
        target = self.targets[index]
        if self.only_features:
            img = self.features[index]
        else:
            img_path = self.data[index]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
        if self.train:
            noisy_label = self.noisy_labels[index]  # same thing as target
            return img, noisy_label, target, index
        return img, target, index

    def __len__(self):
        return len(self.targets)


if __name__ == "__main__":
    from sklearn.neighbors import NearestNeighbors
    import faiss

    save_dir = '../../../data/clothing1m/'

    norm_mean = [0.485, 0.456, 0.406]  # (0.6959, 0.6537, 0.6371)
    norm_std = [0.229, 0.224, 0.225]  # (0.3113, 0.3192, 0.3214)
    transform_train = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=norm_mean, std=norm_std)])
    transform_test = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=norm_mean, std=norm_std)])

    clothing1m_train = Clothing1M(save_dir, train=True, transform=transform_train)
    clothing1m_test = Clothing1M(save_dir, train=False, transform=transform_test)
    print(f"train size - {len(clothing1m_train)}")
    print(f"test size - {len(clothing1m_test)}")

    data = clothing1m_train.normalized_features
    index = faiss.IndexFlatL2(data.shape[1])  # L2 distance
    index.add(data)  # Add data to the index

    # Perform search
    k = 5  # Number of neighbors
    distances, indices = index.search(data, k)
    print(distances.shape)


