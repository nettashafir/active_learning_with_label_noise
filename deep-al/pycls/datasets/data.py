# This file is modified from a code implementation shared with me by Prateek Munjal et al., authors of the paper https://arxiv.org/abs/2002.09564
# GitHub: https://github.com/PrateekMunjal
# ----------------------------------------------------------
import os
import torch
import numpy as np

from torchvision import transforms
from torch.utils.data import Sampler, SubsetRandomSampler

from .randaugment import RandAugmentPolicy
from .simclr_augment import get_simclr_ops
import pycls.utils.logging as lu
from pycls.datasets.custom_datasets import CIFAR10, CIFAR100, MNIST, SVHN
from pycls.datasets.imbalanced_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from pycls.datasets.sampler import IndexedSequentialSampler
from pycls.datasets.tiny_imagenet import TinyImageNet
from pycls.datasets.imagenet import ImageNet, ImageNetSubset
from pycls.datasets.clothing1m import Clothing1M
from pycls.datasets.food101n import Food101N
from pycls.datasets.webvision import WebVision
logger = lu.get_logger(__name__)


class SubsetWeightedRandomSampler(Sampler):
    def __init__(self, indices, weights, replacement=True):
        super().__init__(self)
        self.indices = indices
        # self.weights = torch.as_tensor(weights, dtype=torch.double)[self.indices]
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.replacement = replacement

    def __iter__(self):
        for i in torch.multinomial(self.weights, len(self.indices), self.replacement).tolist():
            yield self.indices[i]
        # return iter(torch.multinomial(self.weights, len(self.indices), self.replacement).tolist())

    def __len__(self):
        return len(self.indices)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class Data:
    """
    Contains all data related functions. For working with new dataset 
    make changes to following functions:
    0. Create labeled.txt and unlabaled.txt for Active Learning
    1. getDataset
    2. getAugmentations
    3. getDataLoaders

    """
    def __init__(self, cfg):
        """
        Initializes dataset attribute of (Data class) object with specified "dataset" argument.
        INPUT:
        cfg: yacs.config, config object
        """
        self.cfg = cfg
        self.num_workers = cfg.DATA_LOADER.NUM_WORKERS
        self.dataset = cfg.DATASET.NAME
        self.data_dir = cfg.DATASET.ROOT_DIR
        # self.target_dir = {"test": cfg.DATASET.TEST_DIR, "train": cfg.DATASET.TRAIN_DIR, "val": cfg.DATASET.VAL_DIR}
        self.aug_method = cfg.DATASET.AUG_METHOD
        self.rand_augment_N = 1 if cfg is None else cfg.RANDAUG.N
        self.rand_augment_M = 5 if cfg is None else cfg.RANDAUG.M

    def about(self):
        """
        Show all properties of this class.
        """
        print(self.__dict__)

    def make_data_lists(self, exp_dir):
        """
        Creates train.txt, test.txt and valid.txt. Text format is chosen to allow readability. 
        Keyword arguments:
            exp_dir -- Full path to the experiment directory where index lists will be saved
        """
        train = os.path.join(exp_dir, 'train.txt')
        test = os.path.join(exp_dir, 'test.txt')
        
        if os.path.exists(train) or os.path.exists(test):
            out = f'train.txt or test.text already exist at {exp_dir}'
            return None
        
        train_list = glob.glob(os.path.join(path, 'train/**/*.png'), recursive=True)
        test_list = glob.glob(os.path.join(path, 'test/**/*.png'), recursive=True)

        with open(train, 'w') as filehandle:
            filehandle.writelines("%s\n" % index for index in train_list)
        
        with open(test, 'w') as filehandle:
            filehandle.writelines("%s\n" % index for index in test_list)

    def getPreprocessOps(self, cfg, train=True):
        """
        This function specifies the steps to be accounted for preprocessing.
        
        INPUT:
        None
        
        OUTPUT:
        Returns a list of preprocessing steps. Note the order of operations matters in the list.
        """
        ops = []
        norm_mean = []
        norm_std = []

        if self.dataset in ["CIFAR10", "CIFAR100", 'IMBALANCED_CIFAR10', 'IMBALANCED_CIFAR100', 'CIFAR10N', 'CIFAR100N']:
            if train:
                ops = [transforms.RandomCrop(32, padding=4)]
            else:
                ops = [transforms.CenterCrop(32)]
            norm_mean = [0.4914, 0.4822, 0.4465]
            norm_std = [0.247 , 0.2435, 0.2616]
        elif self.dataset == "MNIST":
            ops = [transforms.Resize(32)]
            norm_mean = [0.1307,]
            norm_std = [0.3081,]
        elif self.dataset == "TINYIMAGENET":
            if train:
                ops = [transforms.RandomResizedCrop(64, scale=(0.5, 1.))]
            else:
                ops = [transforms.CenterCrop(64)]
            # Using ImageNet values
            norm_mean = [0.485, 0.456, 0.406]
            norm_std = [0.229, 0.224, 0.225]
        elif self.dataset in ["SVHN"]:
            if train:
                ops = [transforms.RandomCrop(32, padding=4)]
            else:
                ops = [transforms.CenterCrop(32)]
            norm_mean = [0.4376, 0.4437, 0.4728]
            norm_std = [0.1980, 0.2010, 0.1970]
        elif self.dataset in ["IMAGENET", 'IMAGENET50', 'IMAGENET100', 'IMAGENET200']:
            if train:
                ops = [transforms.RandomResizedCrop(224, scale=(0.5, 1.))]
            else:
                ops = [transforms.Resize(256), transforms.CenterCrop(224)]
            # Using ImageNet values
            norm_mean = [0.485, 0.456, 0.406]
            norm_std = [0.229, 0.224, 0.225]
        elif self.dataset == "CLOTHING1M":
            if train:
                ops = [transforms.RandomResizedCrop(224, scale=(0.5, 1.0)), transforms.RandomHorizontalFlip(p=0.5)]
            else:
                ops = [transforms.Resize(256), transforms.CenterCrop(224)]
            norm_mean = [0.485, 0.456, 0.406]  # (0.6959, 0.6537, 0.6371)
            norm_std = [0.229, 0.224, 0.225]  # (0.3113, 0.3192, 0.3214)
        elif self.dataset == "FOOD101N":
            if train:
                ops = [transforms.RandomResizedCrop(224, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(p=0.5)]

            else:  # Test/validation transforms
                ops = [
                    # Just resize and center crop for validation/testing
                    transforms.Resize(256),  # Resize shortest side to 256
                    transforms.CenterCrop(224),  # Take center 224x224 crop
                ]
            norm_mean = [0.485, 0.456, 0.406]
            norm_std = [0.229, 0.224, 0.225]
        elif self.dataset in ["WEBVISION", "MINIWEBVISION_FLICKR", "MINIWEBVISION_GOOGLE"]:
            if train:
                ops = [transforms.RandomResizedCrop(224, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(p=0.5)]

            else:  # Test/validation transforms
                ops = [
                    # Just resize and center crop for validation/testing
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                ]
            norm_mean = [0.485, 0.456, 0.406]
            norm_std = [0.229, 0.224, 0.225]
        else:
            raise NotImplementedError

        if train and self.aug_method == 'simclr':
            ops.insert(1, get_simclr_ops(input_shape=cfg.TRAIN.IM_SIZE))

        elif train and self.aug_method == 'randaug':
            #N and M values are taken from Experiment Section of RandAugment Paper
            #Though RandAugment paper works with WideResNet model
            ops.append(RandAugmentPolicy(N=self.rand_augment_N, M=self.rand_augment_M))

        elif train and self.aug_method == 'hflip':
            ops.append(transforms.RandomHorizontalFlip())

        ops.append(transforms.ToTensor())
        ops.append(transforms.Normalize(norm_mean, norm_std))

        return ops

    def getDataset(self, save_dir, isTrain=True, isDownload=False, only_features=False, representation_model="simclr", exp_dir=None, project_features=False):
        """
        This function returns the dataset instance and number of data points in it.
        
        INPUT:
        save_dir: String, It specifies the path where dataset will be saved if downloaded.
        
        preprocess_steps(optional): List, Contains the ordered operations used for preprocessing the data.
        
        isTrain (optional): Bool, If true then Train partition is downloaded else Test partition.
        
        isDownload (optional): Bool, If true then dataset is saved at path specified by "save_dir".

        only_features (optional): Bool, If true then dataset will return SSL features instead of the Image.

        representation_model (optional): String, Specifies the type of representation model to be used for SSL features.

        exp_dir (optional): String, Specifies the path to experiment directory.

        project_features (optional): Bool, If true then features will be projected to the unit sphere.
        
        OUTPUT:
        (On Success) Returns the tuple of dataset instance and length of dataset.
        (On Failure) Returns Message as <dataset> not specified.
        """
        self.eval_mode = True
        test_preops_list = self.getPreprocessOps(self.cfg, train=False)
        test_preprocess_steps = transforms.Compose(test_preops_list)
        self.eval_mode = False
        exp_dir = exp_dir if exp_dir is not None else self.cfg.EXP_DIR
        
        if isTrain:
            preprocess_steps = self.getPreprocessOps(self.cfg, train=True)
        else:
            preprocess_steps = test_preops_list

        split = "Train" if isTrain else "Test"
        print(f"{split} Preprocess Operations Selected ==> {str(preprocess_steps)}")
        # logger.info(f"{split} Preprocess Operations Selected ==> {str(preprocess_steps)}")
        preprocess_steps = transforms.Compose(preprocess_steps)

        if self.dataset == "MNIST":
            mnist = MNIST(save_dir, train=isTrain, transform=preprocess_steps, test_transform=test_preprocess_steps, download=isDownload)
            return mnist, len(mnist)

        # elif self.dataset == "CIFAR10":
        #     cifar10 = CIFAR10(save_dir, train=isTrain, transform=preprocess_steps, test_transform=test_preprocess_steps, download=isDownload, only_features=only_features)
        #     return cifar10, len(cifar10)
        #
        # elif self.dataset == "CIFAR100":
        #     cifar100 = CIFAR100(save_dir, train=isTrain, transform=preprocess_steps,  test_transform=test_preprocess_steps, download=isDownload, only_features=only_features)
        #     return cifar100, len(cifar100)

        elif self.dataset in ["CIFAR10", "CIFAR10N"]:
            noise_root = self.cfg.NOISE.ROOT_NOISE_DIR
            noise_type = self.cfg.NOISE.NOISE_TYPE
            noise_rate = self.cfg.NOISE.NOISE_RATE
            cifar10 = CIFAR10(save_dir, noise_root, noise_type, noise_rate, exp_dir, train=isTrain, transform=preprocess_steps, test_transform=test_preprocess_steps, download=isDownload, only_features=only_features, representation_model=representation_model, project_features=project_features)
            return cifar10, len(cifar10)

        elif self.dataset in ["CIFAR100", "CIFAR100N"]:
            noise_root = self.cfg.NOISE.ROOT_NOISE_DIR
            noise_type = self.cfg.NOISE.NOISE_TYPE
            noise_rate = self.cfg.NOISE.NOISE_RATE
            cifar100 = CIFAR100(self.cfg, save_dir, noise_root, noise_type, noise_rate, exp_dir, train=isTrain, transform=preprocess_steps, test_transform=test_preprocess_steps, download=isDownload, only_features=only_features, representation_model=representation_model, project_features=project_features)
            return cifar100, len(cifar100)

        elif self.dataset == "CLOTHING1M":
            clothing1m = Clothing1M(save_dir, train=isTrain, transform=preprocess_steps, representation_model=representation_model, only_features=only_features)
            return clothing1m, len(clothing1m)

        elif self.dataset == "FOOD101N":
            food101n = Food101N(save_dir, train=isTrain, transform=preprocess_steps, representation_model=representation_model, only_features=only_features)
            return food101n, len(food101n)

        elif self.dataset == "MINIWEBVISION_FLICKR":
            miniwebvision = WebVision(save_dir, train=isTrain, transform=preprocess_steps, representation_model=representation_model, only_features=only_features, num_class=50, image_source="flickr")
            return miniwebvision, len(miniwebvision)

        elif self.dataset == "MINIWEBVISION_GOOGLE":
            miniwebvision = WebVision(save_dir, train=isTrain, transform=preprocess_steps, representation_model=representation_model, only_features=only_features, num_class=50, image_source="google")
            return miniwebvision, len(miniwebvision)

        elif self.dataset == "SVHN":
            if isTrain:
                svhn = SVHN(save_dir, split='train', transform=preprocess_steps,  test_transform=test_preprocess_steps, download=isDownload)
            else:
                svhn = SVHN(save_dir, split='test', transform=preprocess_steps,  test_transform=test_preprocess_steps, download=isDownload)
            return svhn, len(svhn)

        elif self.dataset == "TINYIMAGENET":
            noise_type = self.cfg.NOISE.NOISE_TYPE
            noise_rate = self.cfg.NOISE.NOISE_RATE
            if isTrain:
                # tiny = datasets.ImageFolder(save_dir+'/train', transform=preprocess_steps)
                tiny = TinyImageNet(save_dir, split='train', noise_type=noise_type, noise_rate=noise_rate, exp_dir=exp_dir, transform=preprocess_steps, test_transform=test_preprocess_steps, only_features=only_features, representation_model=representation_model)
            else:
                # tiny = datasets.ImageFolder(save_dir+'/val', transform=preprocess_steps)
                tiny = TinyImageNet(save_dir, split='val', noise_type=noise_type, noise_rate=noise_rate, exp_dir=exp_dir, transform=preprocess_steps, test_transform=test_preprocess_steps, only_features=only_features, representation_model=representation_model)
            return tiny, len(tiny)

        elif self.dataset in ['IMAGENET']:
            if isTrain:
                # tiny = datasets.ImageFolder(save_dir+'/train', transform=preprocess_steps)
                imagenet = ImageNet(save_dir, split='train', transform=preprocess_steps, test_transform=test_preprocess_steps,
                                      num_classes=self.cfg.MODEL.NUM_CLASSES, only_features=only_features)
            else:
                # tiny = datasets.ImageFolder(save_dir+'/val', transform=preprocess_steps)
                imagenet = ImageNet(save_dir, split='val', transform=preprocess_steps, test_transform=test_preprocess_steps,
                                      num_classes=self.cfg.MODEL.NUM_CLASSES, only_features=only_features)
            return imagenet, len(imagenet)

        elif self.dataset in ['IMAGENET50', 'IMAGENET100', 'IMAGENET200']:
            data_set_lables = 50
            if self.dataset == 'IMAGENET100':
                data_set_lables = 100
            elif self.dataset == 'IMAGENET200':
                data_set_lables = 200
            subset_file = os.path.join(save_dir, f'imagenet_{data_set_lables}.txt')
            if isTrain:
                noise_type = self.cfg.NOISE.NOISE_TYPE
                noise_rate = self.cfg.NOISE.NOISE_RATE
                imagenet = ImageNetSubset(self.cfg,
                    subset_file, save_dir, exp_dir, split='train',
                    transform=preprocess_steps, test_transform=test_preprocess_steps, only_features=only_features, representation_model=representation_model, noise_type=noise_type, noise_rate=noise_rate)
            else:
                imagenet = ImageNetSubset(self.cfg,
                    subset_file, save_dir, exp_dir, split='val',
                    transform=preprocess_steps, test_transform=test_preprocess_steps, only_features=only_features, representation_model=representation_model,)
            return imagenet, len(imagenet)

        elif self.dataset == 'IMBALANCED_CIFAR10':
            im_cifar10 = IMBALANCECIFAR10(save_dir, train=isTrain, transform=preprocess_steps, test_transform=test_preprocess_steps)
            return im_cifar10, len(im_cifar10)

        elif self.dataset == 'IMBALANCED_CIFAR100':
            im_cifar100 = IMBALANCECIFAR100(save_dir, train=isTrain, transform=preprocess_steps, test_transform=test_preprocess_steps)
            return im_cifar100, len(im_cifar100)

        else:
            print("Either the specified {} dataset is not added or there is no if condition in getDataset function of Data class".format(self.dataset))
            logger.info("Either the specified {} dataset is not added or there is no if condition in getDataset function of Data class".format(self.dataset))
            raise NotImplementedError

    def makeLUVSets(self, train_split_ratio, val_split_ratio, data, seed_id, save_dir):
        """
        Initialize the labelled and unlabelled set by splitting the data into train
        and validation according to split_ratios arguments.

        Visually it does the following:

        |<------------- Train -------------><--- Validation --->

        |<--- Labelled --><---Unlabelled --><--- Validation --->

        INPUT:
        train_split_ratio: Float, Specifies the proportion of data in train set.
        For example: 0.8 means beginning 80% of data is training data.

        val_split_ratio: Float, Specifies the proportion of data in validation set.
        For example: 0.1 means ending 10% of data is validation data.

        data: reference to dataset instance. This can be obtained by calling getDataset function of Data class.
        
        OUTPUT:
        (On Success) Sets the labelled, unlabelled set along with validation set
        (On Failure) Returns Message as <dataset> not specified.
        """
        # Reproducibility stuff
        torch.manual_seed(seed_id)
        np.random.seed(seed_id)

        assert isinstance(train_split_ratio, float),"Train split ratio is of {} datatype instead of float".format(type(train_split_ratio))
        assert isinstance(val_split_ratio, float),"Val split ratio is of {} datatype instead of float".format(type(val_split_ratio))

        lSet = []
        uSet = []
        valSet = []
        
        n_dataPoints = len(data)
        all_idx = [i for i in range(n_dataPoints)]
        np.random.shuffle(all_idx)
        train_splitIdx = int(train_split_ratio*n_dataPoints)
        #To get the validation index from end we multiply n_datapoints with 1-val_ratio 
        val_splitIdx = int((1-val_split_ratio)*n_dataPoints)
        #Check there should be no overlap with train and val data
        assert train_split_ratio + val_split_ratio < 1.0, "Validation data over laps with train data as last train index is {} and last val index is {}. \
            The program expects val index > train index. Please satisfy the constraint: train_split_ratio + val_split_ratio < 1.0; currently it is {} + {} is not < 1.0 => {} is not < 1.0"\
                .format(train_splitIdx, val_splitIdx, train_split_ratio, val_split_ratio, train_split_ratio + val_split_ratio)
        
        lSet = all_idx[:train_splitIdx]
        uSet = all_idx[train_splitIdx:val_splitIdx]
        valSet = all_idx[val_splitIdx:]

        # print("=============================")
        # print("lSet len: {}, uSet len: {} and valSet len: {}".format(len(lSet),len(uSet),len(valSet)))
        # print("=============================")
        
        lSet = np.array(lSet, dtype=np.ndarray)
        uSet = np.array(uSet, dtype=np.ndarray)
        valSet = np.array(valSet, dtype=np.ndarray)
        
        np.save(f'{save_dir}/lSet.npy', lSet)
        np.save(f'{save_dir}/uSet.npy', uSet)
        np.save(f'{save_dir}/valSet.npy', valSet)
        
        return f'{save_dir}/lSet.npy', f'{save_dir}/uSet.npy', f'{save_dir}/valSet.npy'

    def makeTVSets(self, val_split_ratio, data, seed_id, save_dir):
        """
        Initialize the train and validation sets by splitting the train data according to split_ratios arguments.

        Visually it does the following:

        |<------------- Train -------------><--- Validation --->

        INPUT:
        val_split_ratio: Float, Specifies the proportion of data in validation set.
        For example: 0.1 means ending 10% of data is validation data.

        data: reference to dataset instance. This can be obtained by calling getDataset function of Data class.
        
        OUTPUT:
        (On Success) Sets the train set and the validation set
        (On Failure) Returns Message as <dataset> not specified.
        """
        # Reproducibility stuff
        torch.manual_seed(seed_id)
        np.random.seed(seed_id)

        assert isinstance(val_split_ratio, float),"Val split ratio is of {} datatype instead of float".format(type(val_split_ratio))

        trainSet = []
        valSet = []
        
        n_dataPoints = len(data)
        all_idx = [i for i in range(n_dataPoints)]
        np.random.shuffle(all_idx)

        # To get the validation index from end we multiply n_datapoints with 1-val_ratio 
        val_splitIdx = int((1-val_split_ratio)*n_dataPoints)
        
        trainSet = all_idx[:val_splitIdx]
        valSet = all_idx[val_splitIdx:]

        # print("=============================")
        # print("lSet len: {}, uSet len: {} and valSet len: {}".format(len(lSet),len(uSet),len(valSet)))
        # print("=============================")
        
        trainSet = np.array(trainSet, dtype=np.ndarray)
        valSet = np.array(valSet, dtype=np.ndarray)
        
        np.save(f'{save_dir}/trainSet.npy', trainSet)
        np.save(f'{save_dir}/valSet.npy', valSet)
        
        return f'{save_dir}/trainSet.npy', f'{save_dir}/valSet.npy'

    def makeUVSets(self, val_split_ratio, data, seed_id, save_dir): 
        """
        Initial labeled pool should already be sampled. We use this function to initialize the train and validation sets by splitting the train data according to split_ratios arguments.

        Visually it does the following:

        |<------------- Unlabeled -------------><--- Validation --->

        INPUT:
        val_split_ratio: Float, Specifies the proportion of data in validation set.
        For example: 0.1 means ending 10% of data is validation data.

        data: reference to uSet instance post initial pool sampling. This can be obtained by calling getDataset function of Data class.
        
        OUTPUT:
        (On Success) Sets the unlabeled set and the validation set
        (On Failure) Returns Message as <dataset> not specified.
        """
        # Reproducibility stuff
        torch.manual_seed(seed_id)
        np.random.seed(seed_id)

        assert isinstance(val_split_ratio, float),"Val split ratio is of {} datatype instead of float".format(type(val_split_ratio))
        uSet = []
        valSet = []
        
        n_dataPoints = len(data)
        # all_idx = [i for i in range(n_dataPoints)]
        np.random.shuffle(data)

        # To get the validation index from end we multiply n_datapoints with 1-val_ratio 
        val_splitIdx = int((1-val_split_ratio)*n_dataPoints)
        
        uSet = data[:val_splitIdx]
        valSet = data[val_splitIdx:]

        # print("=============================")
        # print("lSet len: {}, uSet len: {} and valSet len: {}".format(len(lSet),len(uSet),len(valSet)))
        # print("=============================")
        
        uSet = np.array(uSet, dtype=np.ndarray)
        valSet = np.array(valSet, dtype=np.ndarray)
        
        np.save(f'{save_dir}/uSet.npy', uSet)
        np.save(f'{save_dir}/valSet.npy', valSet)
        
        return f'{save_dir}/uSet.npy', f'{save_dir}/valSet.npy'

    def getIndexesDataLoader(self, indexes, batch_size, data):
        """
        Gets reference to the data loader which provides batches of <batch_size> by randomly sampling
        from indexes set. We use SubsetRandomSampler as sampler in returned DataLoader.

        ARGS
        -----

        indexes: np.ndarray, dtype: int, Array of indexes which will be used for random sampling.

        batch_size: int, Specifies the batchsize used by data loader.

        data: reference to dataset instance. This can be obtained by calling getDataset function of Data class.

        OUTPUT
        ------

        Returns a reference to dataloader
        """

        assert isinstance(indexes, np.ndarray), "Indexes has dtype: {} whereas expected is nd.array.".format(type(indexes))
        assert isinstance(batch_size, int), "Batchsize is expected to be of int type whereas currently it has dtype: {}".format(type(batch_size))
        while len(indexes) < batch_size:
            orig_indexes = indexes
            indexes = np.concatenate((indexes, orig_indexes))

        subsetSampler = SubsetRandomSampler(indexes)
        # # print(data)
        # if self.dataset == "IMAGENET":
        #     loader = DataLoader(dataset=data, batch_size=batch_size,sampler=subsetSampler, pin_memory=True)
        # else:
        batch_size = min(batch_size, len(indexes))

        loader = MultiEpochsDataLoader(dataset=data, batch_size=batch_size, num_workers=2*self.num_workers, sampler=subsetSampler, pin_memory=True, drop_last=True)
        return loader

    def getBalancedIndexesDataLoader(self, indexes, batch_size, data):
        assert isinstance(indexes, np.ndarray), "Indexes has dtype: {} whereas expected is nd.array.".format(type(indexes))
        assert isinstance(batch_size, int), "Batchsize is expected to be of int type whereas currently it has dtype: {}".format(type(batch_size))
        while len(indexes) < batch_size:
            orig_indexes = indexes
            indexes = np.concatenate((indexes, orig_indexes))

        labels = torch.Tensor(data.noisy_labels)[indexes].to(torch.int64)
        class_counts = torch.bincount(labels)
        class_counts[class_counts == 0] = 1  # Avoid division by zero
        class_weights = 1. / class_counts.float()
        weights = class_weights[labels]

        subsetSampler = SubsetWeightedRandomSampler(indexes, weights)
        batch_size = min(batch_size, len(indexes))

        loader = MultiEpochsDataLoader(dataset=data, batch_size=batch_size, num_workers=2*self.num_workers, sampler=subsetSampler, pin_memory=True, drop_last=True)
        return loader


    def getSequentialDataLoader(self, indexes, batch_size, data):
        """
        Gets reference to the data loader which provides batches of <batch_size> sequentially 
        from indexes set. We use IndexedSequentialSampler as sampler in returned DataLoader.

        ARGS
        -----

        indexes: np.ndarray, dtype: int, Array of indexes which will be used for random sampling.

        batch_size: int, Specifies the batchsize used by data loader.

        data: reference to dataset instance. This can be obtained by calling getDataset function of Data class.

        OUTPUT
        ------

        Returns a reference to dataloader
        """

        assert isinstance(indexes, np.ndarray), "Indexes has dtype: {} whereas expected is nd.array.".format(type(indexes))
        assert isinstance(batch_size, int), "Batchsize is expected to be of int type whereas currently it has dtype: {}".format(type(batch_size))
        #
        # while len(indexes) < batch_size:
        #     orig_indexes = indexes
        #     indexes = np.concatenate((indexes, orig_indexes))
        
        subsetSampler = IndexedSequentialSampler(indexes)
        # if self.dataset == "IMAGENET":
        #     loader = DataLoader(dataset=data, batch_size=batch_size,sampler=subsetSampler,pin_memory=True)
        # else:

        loader = MultiEpochsDataLoader(dataset=data, num_workers=self.num_workers, batch_size=batch_size, sampler=subsetSampler, shuffle=False, pin_memory=True)
        return loader


    def getTestLoader(self, data, test_batch_size, seed_id=0):
        """
        Implements a random subset sampler for sampling the data from test set.
        
        INPUT:
        data: reference to dataset instance. This can be obtained by calling getDataset function of Data class.
        
        test_batch_size: int, Denotes the size of test batch

        seed_id: int, Helps in reporoducing results of random operations
        
        OUTPUT:
        (On Success) Returns the testLoader
        (On Failure) Returns Message as <dataset> not specified.
        """
        # Reproducibility stuff
        torch.manual_seed(seed_id)
        np.random.seed(seed_id)

        n_datapts = len(data)
        idx = [i for i in range(n_datapts)]
        #np.random.shuffle(idx)

        test_sampler = SubsetRandomSampler(idx)
        testLoader = MultiEpochsDataLoader(data, num_workers=self.num_workers, batch_size=test_batch_size, sampler=test_sampler, pin_memory=True)
        return testLoader


    def loadPartitions(self, lSetPath, uSetPath, valSetPath):

        assert isinstance(lSetPath, str), "Expected lSetPath to be a string."
        assert isinstance(uSetPath, str), "Expected uSetPath to be a string."
        assert isinstance(valSetPath, str), "Expected valSetPath to be a string."

        lSet = np.load(lSetPath, allow_pickle=True)
        uSet = np.load(uSetPath, allow_pickle=True)
        valSet = np.load(valSetPath, allow_pickle=True)

        #Checking no overlap
        assert len(set(valSet) & set(uSet)) == 0, "Intersection is not allowed between validationset and uset"
        assert len(set(valSet) & set(lSet)) == 0, "Intersection is not allowed between validationset and lSet"
        assert len(set(uSet) & set(lSet)) == 0, "Intersection is not allowed between uSet and lSet"

        return lSet, uSet, valSet

    def loadTVPartitions(self, trainSetPath, valSetPath):

        assert isinstance(trainSetPath, str), "Expected trainSetPath to be a string."
        assert isinstance(valSetPath, str), "Expected valSetPath to be a string."

        trainSet = np.load(trainSetPath, allow_pickle=True)
        valSet = np.load(valSetPath, allow_pickle=True)

        #Checking no overlap
        assert len(set(valSet) & set(trainSet)) == 0,"Intersection is not allowed between validationset and trainSet"

        return trainSet, valSet


    def loadPartition(self, setPath):

        assert isinstance(setPath, str), "Expected setPath to be a string."

        setArray = np.load(setPath, allow_pickle=True)
        return setArray


    def saveSets(self, lSet, uSet, activeSet, save_dir):

        lSet = np.array(lSet, dtype=np.ndarray)
        uSet = np.array(uSet, dtype=np.ndarray)
        valSet = np.array(activeSet, dtype=np.ndarray)

        np.save(f'{save_dir}/lSet.npy', lSet)
        np.save(f'{save_dir}/uSet.npy', uSet)
        np.save(f'{save_dir}/activeSet.npy', activeSet)
        print(f"Data Object | Saved lSet, uSet and activeSet at {save_dir}")

        # return f'{save_dir}/lSet.npy', f'{save_dir}/uSet.npy', f'{save_dir}/activeSet.npy'


    def saveSet(self, setArray, setName, save_dir):

        setArray = np.array(setArray, dtype=np.ndarray)
        np.save(f'{save_dir}/{setName}.npy', setArray)
        print(f"Data Object | Saved {setName} at {save_dir}")
        # return f'{save_dir}/{setName}.npy'


    def getClassWeightsFromDataset(self, dataset, index_set, bs):
        temp_loader = self.getIndexesDataLoader(indexes=index_set, batch_size=bs, data=dataset)
        return self.getClassWeights(temp_loader)


    def getClassWeights(self, dataloader):

        """
        INPUT
        dataloader: dataLoader
        
        OUTPUT
        Returns a tensor of size C where each element at index i represents the weight for class i. 
        """

        all_labels = []
        for _,y in dataloader:
            all_labels.append(y)
        print("===Computing Imbalanced Weights===")
        
        
        all_labels = np.concatenate(all_labels, axis=0)
        print(f"all_labels.shape: {all_labels.shape}")
        classes = np.unique(all_labels)
        print(f"classes: {classes.shape}")
        num_classes = len(classes)
        freq_count = np.zeros(num_classes, dtype=int)
        for i in classes:
            freq_count[i] = (all_labels==i).sum()
        
        #Normalize
        freq_count = (1.0*freq_count)/np.sum(freq_count)
        print(f"=== Sum(freq_count): {np.sum(freq_count)} ===")
        class_weights = 1./freq_count
        
        class_weights = torch.Tensor(class_weights)
        return class_weights