import os
import sys
import random
from PIL import Image
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


try:
    from pycls.core.builders import build_model
except ImportError:
    add_path(os.path.abspath('../..'))
    from pycls.core.builders import build_model

try:
    from .lnl_utils import get_indexes_data_loader, EnsembleNet, BetaMixture1D
except ImportError:
    from lnl_utils import get_indexes_data_loader, EnsembleNet, BetaMixture1D

DisagreeNet = "disagreenet"


# -------------------------------- utils --------------------------------

class TrainDataset(Dataset):
    """
    A Dataset wrapper used to identify noisy data.

    Samples are returned as (x, y, index), and a subset of samples are returned with a new, fake label
    instead of their original label.
    """

    def __init__(self, dataset, linear_from_features=True):
        self.dataset = dataset
        self.linear_from_features = linear_from_features

    def __getitem__(self, index):
        noisy_label = self.dataset.noisy_labels[index]

        if self.linear_from_features:
            img = self.dataset.features[index]
        else:
            img = self.dataset.data[index]
            img = Image.fromarray(img)
            if self.dataset.transform is not None:
                img = self.dataset.transform(img)

        return img, noisy_label, index

    def __len__(self):
        return len(self.dataset)


# -------------------------------- MAP --------------------------------

class MAP:
    def __init__(self,
                 train_data,
                 l_set,
                 u_set,
                 cfg,
                 batch_size             = 64,
                 max_epochs             = 300,
                 agreement_threshold    = 0.9,
                 num_models             = 10,
                 num_checkpoints        = 8,
                 linear_from_features   = True,
                 filtering_criterion    = DisagreeNet
                 ):
        """
        Initializes the MAP class for training an ensemble of models and applying the MAP / DisagreeNet methods.

        MAP Described in Stern et al. (2023) https://arxiv.org/abs/2310.11077
        DisagreeNet Described in Shwartz et al. (2022) https://arxiv.org/abs/2210.00583

        Parameters:
        train_data (tuple): An object containing the training data and labels.
        l_set (np.ndarray): The labeled set indices.
        u_set (np.ndarray): The unlabeled set indices.
        cfg (object): Configuration object, including model, number of epochs, save interval, etc.
        """
        self.l_set = l_set
        self.u_set = u_set
        self.cfg = cfg
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        # frac = len(self.l_set) / (len(self.l_set) + len(self.u_set))
        # self.agreement_threshold = (1 - frac) * agreement_threshold
        self.agreement_threshold = agreement_threshold
        self.num_models = num_models
        self.num_checkpoints = num_checkpoints
        self.filtering_criterion = filtering_criterion
        self.optimizer_params = {"lr": 0.01, "weight_decay": 5e-4, "momentum": 0.9, "nesterov": True}
        self.linear_from_features = linear_from_features

        train_data = TrainDataset(train_data, linear_from_features=self.linear_from_features)
        all_train_predictions, ELPs, test_models, final_mean_epoch_loss = self.train_ensemble(train_data, l_set)

        self.all_train_predictions = all_train_predictions.astype(int)
        self.ELPs = ELPs.astype(float)
        if self.filtering_criterion == DisagreeNet:
            print("DisagreeNet | Fitting Beta Mixture Model")
            l_set_ELPs = np.take(ELPs, l_set.astype(int))
            bmm = BetaMixture1D()
            # bmm.fit(l_set_ELPs)
            bmm.fit(ELPs)
            self.l_set_is_clean = bmm.predict(l_set_ELPs)
        else:
            self.l_set_is_clean = np.take(self.all_train_predictions, l_set.astype(int)) == np.take(train_data.dataset.noisy_labels, l_set.astype(int))

        self.final_train_loss = final_mean_epoch_loss
        self.models = test_models
        self.scores = l_set_ELPs

    def train_ensemble(self, train_data, train_indices):
        """
        Train the ensemble of models and stop when all networks agree on most samples.
        """
        print("MAP | Build Models")
        train_loader = get_indexes_data_loader(indices=self.l_set,
                                               batch_size=self.batch_size,
                                               data=train_data,
                                               num_workers=self.cfg.DATA_LOADER.NUM_WORKERS)
        models = []
        for i in range(self.num_models):
            model = build_model(self.cfg, linear_from_features=self.linear_from_features).cuda()
            models.append(model)
        checkpoints = [[] for _ in range(self.num_models)]
        optimizers = [SGD(model.parameters(), **self.optimizer_params) for model in models]
        schedulers = [lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs) for optimizer in optimizers]
        criterion = nn.CrossEntropyLoss()
        mean_epoch_loss = None

        # Training loop
        print("MAP | Beginning training")
        for epoch in range(self.max_epochs):
            for model in models:
                model.train()

            mean_epoch_loss = 0
            for batch_idx, (imgs, noisy_labels, _) in enumerate(train_loader):
                imgs, noisy_labels = imgs.cuda(), noisy_labels.cuda()
                for model, optimizer in zip(models, optimizers):
                    optimizer.zero_grad()
                    output = model(imgs)
                    loss = criterion(output, noisy_labels)
                    loss.backward()
                    optimizer.step()
                    mean_epoch_loss += loss.item()

            # Scheduler step after each epoch for each model's optimizer
            for scheduler in schedulers:
                scheduler.step()
            mean_epoch_loss /= (len(train_indices) * self.num_models)
            agreement = self._check_agreement(models, train_loader)
            print(f"MAP | Epoch [{epoch}/{self.max_epochs}], Loss: {mean_epoch_loss:.4f}, Agreement: {agreement * 100:.2f}%")
            self._save_model_checkpoints(models, checkpoints)
            if agreement > self.agreement_threshold:
                print(f"MAP | Stopping early at epoch {epoch} due to high agreement among networks.")
                break

        # make predictions for all the train date
        if self.filtering_criterion == "disagreenet":
            print("DisagreeNet | Get ELPs for all train data")
        else:
            print("MAP | Get predictions for all train data")

        with torch.no_grad():
            all_train_predictions = []
            ELPs = []
            ensemble_model = EnsembleNet(checkpoints)
            ensemble_model.eval()
            all_train_data_loader = DataLoader(train_data, batch_size=500, shuffle=False, num_workers=self.cfg.DATA_LOADER.NUM_WORKERS)
            for imgs, noisy_labels, indices in tqdm(all_train_data_loader):
                imgs, noisy_labels = imgs.cuda(), noisy_labels.cuda()

                if self.filtering_criterion == "disagreenet":
                    elp = ensemble_model.elp(imgs, noisy_labels)
                    ELPs = np.concatenate((ELPs, elp.cpu().numpy()))
                else:
                    predictions = ensemble_model(imgs)
                    all_train_predictions = np.concatenate((all_train_predictions, predictions.cpu().numpy()))

            all_train_predictions = np.asarray(all_train_predictions)
            ELPs = np.asarray(ELPs)

        # save the final number of checkpoints for inference of test data
        print("MAP | Prune checkpoints")
        test_models = []
        return all_train_predictions, ELPs, test_models, mean_epoch_loss

    def _save_model_checkpoints(self, models, checkpoints):
        assert len(models) == len(checkpoints), "Number of models and checkpoints lists should be equal."
        for i, model in enumerate(models):
            model_copy = build_model(self.cfg, linear_from_features=self.linear_from_features).cuda()
            model_copy.load_state_dict(model.state_dict())
            model_copy.eval()
            checkpoints[i].append(model_copy)

    def _prune_checkpoints(self, model_checkpoints):
        pruned_model_checkpoints = []
        epochs = len(model_checkpoints)
        jumps = epochs // self.num_checkpoints
        for i in range(jumps // 2, epochs, jumps):
            pruned_model_checkpoints.append(model_checkpoints[i])
        return pruned_model_checkpoints

    @staticmethod
    def _check_agreement(models, train_loader):
        """
        Check the agreement rate among all models on the training dataset.

        Returns:
        float: The fraction of samples where all models agree.
        """
        all_predictions = []
        for model in models:
            model.eval()
        with torch.no_grad():
            for data, _, _ in train_loader:
                data = data.cuda()
                model_preds = []
                for model in models:
                    outputs = model(data)
                    preds = torch.argmax(outputs, dim=1)
                    model_preds.append(preds.unsqueeze(1))
                all_predictions.append(torch.cat(model_preds, dim=1))

        all_predictions = torch.cat(all_predictions, dim=0)

        # Check how many predictions are identical across all models
        consensus = torch.eq(all_predictions, all_predictions[:, 0].unsqueeze(1)).all(dim=1)
        agreement = consensus.float().mean().item()
        return agreement


# -------------------------------- main - for tests --------------------------------

def get_budget(noise_ratio, samples_per_class, num_classes):
  budget = int(np.ceil((samples_per_class * num_classes) / (1-noise_ratio)))
  return budget


def seed_everything(seed):
    random.seed(seed)  # Python random module.
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU.


def calculate_metrics(true_indices, predicted_indices):
    true_positives = np.sum(np.logical_and(true_indices == 1, predicted_indices == 1))
    false_positives = np.sum(np.logical_and(true_indices == 0, predicted_indices == 1))
    false_negatives = np.sum(np.logical_and(true_indices == 1, predicted_indices == 0))

    true_ratio = np.sum(true_indices) / len(true_indices)                   # (tp + fn) / (tp + fn + tn + fp)
    predicted_ratio = np.sum(predicted_indices) / len(predicted_indices)    # (tp + fp) / (tp + fn + tn + fp)
    recall = true_positives / (true_positives + false_negatives)            # tp / (tp + fn)
    precision = true_positives / (true_positives + false_positives)         # tp / (tp + fp)
    f1_score = 2 * (precision * recall) / (precision + recall)              # tp / (tp + 0.5 * (fp + fn))
    return true_ratio, predicted_ratio, recall, precision, f1_score


if __name__ == "__main__":
    from pycls.core.config import cfg
    from pycls.datasets.data import Data
    from pycls.al.prob_cover import ProbCover

    seed = 0
    dataset_name = 'cifar100'
    noise_mode = 'sym'
    noise_rate = 0.5
    spc = 5
    al_alg = "probcover"

    seed_everything(seed)

    # cfg_file = f"/cs/labs/daphna/nettashaf/TypiClustNoisy/deep-al/configs/{dataset_name}n/al/RESNET18.yaml"
    cfg_file = f"../../configs/{dataset_name}n/al/RESNET18.yaml"
    cfg.merge_from_file(cfg_file)
    cfg.NOISE.ROOT_NOISE_DIR = "/cs/labs/daphna/nettashaf/TypiClustNoisy/cifar-10-100n/data"
    cfg.NOISE.NOISE_TYPE = noise_mode
    cfg.NOISE.NOISE_RATE = noise_rate
    cfg.DATA_LOADER.NUM_WORKERS = 0
    data_obj = Data(cfg)
    train_data, train_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=True, isDownload=True)
    test_data, test_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=False, isDownload=True)

    num_classes = 10 if dataset_name == 'cifar10' else 100
    initial_delta = 0.75 if dataset_name == 'cifar10' else 0.65
    budget = get_budget(noise_rate, spc, num_classes)
    print(f"SPC: {spc}, Budget: {budget}")

    if al_alg == "random":
        l_set = np.random.choice(np.arange(50000), budget, replace=False).astype(int)
        u_set = np.setdiff1d(np.arange(50000), l_set).astype(int)
    elif al_alg == "probcover":
        probcov = ProbCover(cfg, np.array([]), np.arange(50000), budgetSize=budget, delta=initial_delta)
        l_set, u_set = probcov.select_samples()
    else:
        raise NotImplementedError

    is_clean_true = np.asarray(train_data.targets) == train_data.noisy_labels

    map = MAP(train_data, l_set, u_set, cfg,)

    # all_predictions = map.all_train_predictions
    # is_clean_predicted = np.asarray(map.all_train_predictions) == train_data.noisy_labels

    ELPs = map.ELPs
    is_clean_predicted = np.zeros(len(ELPs))
    ELPs_l_set = np.take(ELPs, l_set)
    ELPs_u_set = np.take(ELPs, u_set)

    bmm = BetaMixture1D()
    bmm.fit(ELPs, plot=True)
    is_clean_predicted[l_set] = bmm.predict(ELPs_l_set)
    is_clean_predicted[u_set] = bmm.predict(ELPs_u_set)

    is_clean_predicted_l_set = np.take(is_clean_predicted, l_set)
    is_clean_true_l_set = np.take(is_clean_true, l_set)
    clean_ratio_l_set_true, clean_ratio_l_set_predicted, clean_recall_l_set, clean_precision_l_set, clean_f1_score_l_set \
        = calculate_metrics(is_clean_true_l_set, is_clean_predicted_l_set)
    print(f"l_set:"
          f"\n\tClean ratio true: {clean_ratio_l_set_true}"
          f"\n\tClean ratio predicted: {clean_ratio_l_set_predicted}"
          f"\n\tClean recall: {clean_recall_l_set}"
          f"\n\tClean precision: {clean_precision_l_set}"
          f"\n\tClean F1 score: {clean_f1_score_l_set}")

    is_clean_predicted_u_set = np.take(is_clean_predicted, u_set)
    is_clean_true_u_set = np.take(is_clean_true, u_set)
    clean_ratio_u_set_true, clean_ratio_u_set_predicted, clean_recall_u_set, clean_precision_u_set, clean_f1_score_u_set \
        = calculate_metrics(is_clean_true_u_set, is_clean_predicted_u_set)
    print(f"u_set:"
          f"\n\tClean ratio true: {clean_ratio_u_set_true}"
          f"\n\tClean ratio predicted: {clean_ratio_u_set_predicted}"
          f"\n\tClean recall: {clean_recall_u_set}"
          f"\n\tClean precision: {clean_precision_u_set}"
          f"\n\tClean F1 score: {clean_f1_score_u_set}")


