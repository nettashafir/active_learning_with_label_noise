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


# class clothing_dataloader():
#     def __init__(self, root, batch_size, num_batches, num_workers):
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.num_batches = num_batches
#         self.root = root
#
#         self.transform_train = transforms.Compose([
#             transforms.Resize(256),
#             transforms.RandomCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
#         ])
#         self.transform_test = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
#         ])
#     def run(self,mode,pred=[],prob=[],paths=[]):
#         if mode=='warmup':
#             warmup_dataset = clothing_dataset(self.root,transform=self.transform_train, mode='all',num_samples=self.num_batches*self.batch_size*2)
#             warmup_loader = DataLoader(
#                 dataset=warmup_dataset,
#                 batch_size=self.batch_size*2,
#                 shuffle=True,
#                 num_workers=self.num_workers)
#             return warmup_loader
#         elif mode=='train':
#             labeled_dataset = clothing_dataset(self.root,transform=self.transform_train, mode='labeled',pred=pred, probability=prob,paths=paths)
#             labeled_loader = DataLoader(
#                 dataset=labeled_dataset,
#                 batch_size=self.batch_size,
#                 shuffle=True,
#                 num_workers=self.num_workers)
#             unlabeled_dataset = clothing_dataset(self.root,transform=self.transform_train, mode='unlabeled',pred=pred, probability=prob,paths=paths)
#             unlabeled_loader = DataLoader(
#                 dataset=unlabeled_dataset,
#                 batch_size=int(self.batch_size),
#                 shuffle=True,
#                 num_workers=self.num_workers)
#             return labeled_loader,unlabeled_loader
#         elif mode=='eval_train':
#             eval_dataset = clothing_dataset(self.root,transform=self.transform_test, mode='all',num_samples=self.num_batches*self.batch_size)
#             eval_loader = DataLoader(
#                 dataset=eval_dataset,
#                 batch_size=self.batch_size,
#                 shuffle=False,
#                 num_workers=self.num_workers)
#             return eval_loader
#         elif mode=='test':
#             test_dataset = clothing_dataset(self.root,transform=self.transform_test, mode='test')
#             test_loader = DataLoader(
#                 dataset=test_dataset,
#                 batch_size=1000,
#                 shuffle=False,
#                 num_workers=self.num_workers)
#             return test_loader
#         elif mode=='val':
#             val_dataset = clothing_dataset(self.root,transform=self.transform_test, mode='val')
#             val_loader = DataLoader(
#                 dataset=val_dataset,
#                 batch_size=1000,
#                 shuffle=False,
#                 num_workers=self.num_workers)
#             return val_loader

# p = '/cs/snapless/daphna/daniels44/BiModal/data/logits_ResNet_big50_label_noise_0_clothing_lr_0.01_opt_sgd_reg_0.0005_aug_results/'
# data = np.load(p+'train score mat,pid=0, logits_ResNet_big50_label_noise_0_clothing_lr_0.01_opt_sgd_reg_0.0005_aug=50.npy')
# from scipy.special import softmax
# # ss = softmax(data,axis=2)
# trainset = clothing_dataset(root='/cs/labs/daphna/daphna/data/clothing1m', transform = None)
# targets = trainset.targets
#
# accessibility = (np.argmax(data,axis=2)==targets)*1


if __name__ == "__main__":
    from sklearn.neighbors import NearestNeighbors
    import faiss

    save_dir = '/cs/labs/daphna/data/clothing1m/'

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


