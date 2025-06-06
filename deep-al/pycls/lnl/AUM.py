import os
import sys
import random
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.models import resnet18, resnet34

from aum import AUMCalculator

from pycls.core.builders import FeaturesNet
try:
    from .lnl_utils import get_indexes_data_loader, EnsembleNet, BetaMixture1D, find_nearest_neighbors_set
    import pycls.utils.logging as lu
except ImportError:
    from lnl_utils import get_indexes_data_loader, EnsembleNet, BetaMixture1D, find_nearest_neighbors_set
    import pycls.utils.logging as lu

logger = lu.get_logger(__name__)


# -------------------------------- utils --------------------------------

class ThresholdSamplesDataset(Dataset):
    """
    A Dataset wrapper used to identify noisy data.

    Samples are returned as (x, y, index), and a subset of samples are returned with a new, fake label
    instead of their original label.
    """

    def __init__(self, dataset, threshold_set, linear_from_features=True):
        if not hasattr(dataset, "classes"):
            raise ValueError("dataset must have 'classes' attribute.")

        self.dataset = dataset
        self.threshold_sample_flags = np.zeros(len(dataset), dtype=bool)
        self.threshold_sample_flags[threshold_set] = 1
        self.linear_from_features = linear_from_features
        self.classes = dataset.classes + ["fake_label"]

    def __getitem__(self, index):
        # img, noisy_label, clean_label, index = self.dataset[index]
        # if self.threshold_sample_flags[index]:
        #     noisy_label = len(self.dataset.classes)
        # return img, noisy_label, index

        # Extract the image
        if self.linear_from_features:
            img = self.dataset.features[index]
        else:
            img = self.dataset.data[index]
            img = Image.fromarray(img)
            if self.dataset.transform is not None:
                img = self.dataset.transform(img)

        # Extract the noisy label
        if self.threshold_sample_flags[index]:
            noisy_label = len(self.dataset.classes)
        else:
            noisy_label = self.dataset.noisy_labels[index]

        return img, noisy_label, index

    def __len__(self):
        return len(self.dataset)


# -------------------------------- AUM --------------------------------

class AUM:

    def __init__(self,
                 train_data,
                 l_set,
                 u_set,
                 cfg,
                 num_repeats           = 1,
                 min_threshold_samples = 1,
                 thr_percentile        = 80,
                 epochs                = 40,
                 batch_size            = 64,
                 linear_from_features  = True,
                 test_data             = None,
                 ):
        """
        Initializes the AUM class for training a model and applying the AUM method.

        Described in Pleiss et al. (2020) https://arxiv.org/abs/2001.10528

        Parameters:
        train_data (tuple): An object containing the training data and labels.
        l_set (np.ndarray): The labeled set indices.
        u_set (np.ndarray): The unlabeled set indices.
        cfg (object): Configuration object, including model, number of epochs, save interval, etc.
        """
        self.cfg = cfg
        self.min_threshold_samples = min_threshold_samples
        self.l_set = l_set
        self.thr_percentile = thr_percentile
        self.epochs = epochs
        self.num_repeats = num_repeats
        self.batch_size = batch_size
        self.linear_from_features = linear_from_features
        self.results = None
        self.models = []

        if test_data is not None:
            self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=cfg.DATA_LOADER.NUM_WORKERS)
            self.calculate_test_accuracies = True
        else:
            self.test_loader = None
            self.calculate_test_accuracies = False
        self.test_accuracies = []

        threshold_sets = self._get_threshold_sets(cfg.MODEL.NUM_CLASSES, u_set, train_data.features)
        print(f'AUM | Number of threshold samples: {len(threshold_sets[0])}')
        logger.info(f'AUM | Number of threshold samples: {len(threshold_sets[0])}')

        l_set_is_clean_repeats = []
        scores_repeats = []
        confidence_scores_repeats = []
        for i in range(len(threshold_sets)):
            print(f"AUM | Calculating AUM for threshold set {i + 1}")
            logger.info(f"AUM | Calculating AUM for threshold set {i + 1}")
            train_indices = np.concatenate((self.l_set, threshold_sets[i]))
            aum_calculator = AUMCalculator(save_dir=None)
            threshold_train_data = ThresholdSamplesDataset(train_data, threshold_sets[i], self.linear_from_features)
            train_loader = get_indexes_data_loader(indices=train_indices, batch_size=self.batch_size,
                                                   data=threshold_train_data, num_workers=cfg.DATA_LOADER.NUM_WORKERS)
            model = self.train_model(train_loader, aum_calculator, train_indices)
            self.models.append(model)
            if self.calculate_test_accuracies:
                break  # we only need to calculate test accuracies once
            l_set_is_clean, scores, confidence_scores = self.calculate_l_set_is_clean(train_indices)
            l_set_is_clean_repeats.append(l_set_is_clean)
            scores_repeats.append(scores)
            confidence_scores_repeats.append(confidence_scores)

        # calculate the final clean indices
        self.l_set_is_clean = np.all(l_set_is_clean_repeats, axis=0)
        self.scores = np.mean(scores_repeats, axis=0)
        self.confidence_scores = np.mean(confidence_scores_repeats, axis=0)

    def _get_threshold_sets(self, num_classes, u_set, features):
        num_of_threshold_samples = int(np.ceil(len(self.l_set) / num_classes))

        if self.cfg.NOISE.NEIGHBORS_FOR_THRESHOLD:
            print("AUM | Selecting threshold samples with neighbors")
            logger.info("AUM | Selecting threshold samples with neighbors")
            pool = find_nearest_neighbors_set(features, self.l_set)
        else:
            print("AUM | Selecting threshold samples by random selection")
            logger.info("AUM | Selecting threshold samples by random selection")
            pool = u_set

        threshold_sets = []
        for i in range(self.num_repeats):
            threshold_sets.append(np.random.choice(pool, num_of_threshold_samples, replace=False).astype(int))

        return threshold_sets

    def train_model(self, train_loader, aum_calculator, train_indices):
        # Initialize the model and optimizer
        if self.linear_from_features:
            model = FeaturesNet(self.cfg.DATASET.REPRESENTATION_DIM, self.cfg.MODEL.NUM_CLASSES + 1).cuda()
            # model = model_builder.build_model(self.cfg, num_classes=self.cfg.MODEL.NUM_CLASSES + 1, linear_from_features=True).cuda()
        else:
            model = resnet34(num_classes=self.cfg.MODEL.NUM_CLASSES + 1).cuda()
        optimizer = SGD(model.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9, nesterov=True)
        criterion = nn.CrossEntropyLoss()

        # Training loop with AUM calculations
        print('AUM | Beginning training for AUM calculation')
        logger.info('AUM | Beginning training for AUM calculation')
        for epoch in range(self.epochs):  # Train for the specified number of epochs
            model.train()
            epoch_loss = 0
            for i, (images, noisy_labels, sample_ids) in enumerate(train_loader):
                images, noisy_labels = images.cuda(), noisy_labels.cuda()  # Move data to GPU if available

                # Backward and optimize
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, noisy_labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                # Update AUM for each sample
                aum_calculator.update(outputs, noisy_labels, sample_ids.to(torch.int64).tolist())

            epoch_loss /= len(train_indices)
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(f'AUM | Epoch [{epoch + 1}/{self.epochs}], Loss: {epoch_loss:.4f}')
                logger.info(f'AUM | Epoch [{epoch + 1}/{self.epochs}], Loss: {epoch_loss:.4f}')

            if self.calculate_test_accuracies:
                test_accuracy = self.test_model(model)
                print(f"AUM | test accuracy - {test_accuracy}")
                logger.info(f"AUM | test accuracy - {test_accuracy}")
                self.test_accuracies.append(test_accuracy)

        # Complete Missing Samples
        missing = list(set(train_indices) - set(aum_calculator.counts.keys()))
        if len(missing) > 0:
            print(f'AUM | Completing missing samples: {len(missing)}')
            logger.info(f'AUM | Completing missing samples: {len(missing)}')
            subset = Subset(train_loader.dataset, missing)
            subset_loader = DataLoader(subset, batch_size=len(missing), shuffle=False)
            for images, noisy_labels, sample_ids in subset_loader:
                images, noisy_labels = images.cuda(), noisy_labels.cuda()
                outputs = model(images.cuda())
                aum_calculator.update(outputs, noisy_labels, sample_ids.to(torch.int64).tolist())

        # Finalize aum calculator
        # aum_calculator.finalize()
        results = [{
            'sample_id': sample_id,
             'aum': aum_calculator.sums[sample_id] / aum_calculator.counts[sample_id]
        }
            for sample_id in aum_calculator.counts.keys()]
        result_df = pd.DataFrame(results).sort_values(by='aum', ascending=False)
        self.results = result_df
        return model

    def test_model(self, model):
        if self.test_loader is None:
            print("AUM | No test data available. Skipping testing.")
            logger.info("AUM | No test data available. Skipping testing.")
            return
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels, _ in self.test_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return accuracy

    def calculate_l_set_is_clean(self, train_indices):
        assert self.results is not None, 'AUM | Results are not available. Please train the model first.'
        print('AUM | Calculate AUM threshold and filter noisy samples')
        logger.info('AUM | Calculate AUM threshold and filter noisy samples')

        # Load AUM results
        # aum_records = pd.read_csv(f'{self.output_dir}/aum_values.csv')
        # aums = aum_records["aum"].values
        # indices_perm = aum_records["sample_id"].values

        aums = self.results["aum"].values
        indices_perm = self.results["sample_id"].values

        # Sort the AUM values according to the order of the train indices
        perm_index = {value: index for index, value in enumerate(indices_perm)}
        permutation = [perm_index[value] for value in train_indices]
        aums = aums[permutation]

        # Determine the relevant percentile threshold
        threshold_aums = aums[len(self.l_set):]
        threshold = np.percentile(threshold_aums, self.thr_percentile)

        # Filter out noisy samples
        l_set_aums = aums[:len(self.l_set)]
        l_set_is_clean = threshold <= l_set_aums

        # Calculate confidence scores separately for predicted clean and noisy
        confidence_scores = np.zeros(len(l_set_aums))

        # For predicted clean samples (AUM <= threshold):
        clean_mask = l_set_is_clean
        if np.any(clean_mask):
            clean_aums = l_set_aums[clean_mask]
            # Use relative distance from minimum value to threshold
            confidence_scores[clean_mask] = (clean_aums - threshold) / (np.max(clean_aums) - threshold)

        # For predicted noisy samples (AUM > threshold):
        noisy_mask = ~l_set_is_clean
        if np.any(noisy_mask):
            noisy_aums = l_set_aums[noisy_mask]
            # Use relative distance from threshold to maximum value
            confidence_scores[noisy_mask] = (threshold - noisy_aums) / (threshold - np.min(noisy_aums))

        # #  Remove output directory
        # os.system(f'rm -r {self.output_dir}')

        return l_set_is_clean, l_set_aums, confidence_scores
