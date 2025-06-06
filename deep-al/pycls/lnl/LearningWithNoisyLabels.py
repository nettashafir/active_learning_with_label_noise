import random
import numpy as np
import torch
import pickle

try:
    from .lnl_utils import get_indexes_data_loader, EnsembleNet, BetaMixture1D, find_nearest_neighbors_set
    import pycls.utils.logging as lu
except ImportError:
    from lnl_utils import get_indexes_data_loader, EnsembleNet, BetaMixture1D, find_nearest_neighbors_set
    import pycls.utils.logging as lu

logger = lu.get_logger(__name__)


class LearningWithNoisyLabels:
    def __init__(self, train_data, test_data, cfg):
        self.train_data = train_data
        self.test_data = test_data
        self.cfg = cfg
        if hasattr(self.train_data, 'is_noisy'):
            self.is_clean = np.logical_not(self.train_data.is_noisy)
        else:
            self.is_clean = np.asarray(self.train_data.noisy_labels) == np.asarray(self.train_data.targets)  # for tests and ideal denoiser
        self.noise_records = {}  # {i: 0 for i in range(len(self.train_data))}
        self.num_predictions = {}  # {i: 0 for i in range(len(self.train_data))}

        # For inference
        self.lnl_models = []
        self.all_predictions = []

    @property
    def inference_available(self):
        return len(self.lnl_models) > 0 or len(self.all_predictions) > 0

    def _reset_lnl_models(self):
        if len(self.lnl_models) > 0:
            for model in self.lnl_models:
                del model
            del self.lnl_models

        self.lnl_models = []
        self.all_predictions = []

    def identify_noisy_samples(self, l_set, u_set, return_scores=False, silent=False):
        lnl_method = self.cfg.NOISE.FILTERING_FN.lower()

        if lnl_method == 'ideal':
            if not silent:
                print(f"LNL Object | Predicting noisy samples with ideal denoiser")
                logger.info(f"LNL Object | Predicting noisy samples with ideal denoiser")
            l_set_is_clean = np.take(self.is_clean, l_set.astype(int))
            for i, idx in enumerate(l_set):
                self.noise_records[idx] = l_set_is_clean[i]
            if return_scores:
                return l_set_is_clean, l_set_is_clean, None, None, None
            return l_set_is_clean, None, None

        model = None
        train_loss = None
        scores = None
        confidence_scores = None

        if not silent:
            print(f"LNL Object | Predicting noisy samples by training LNL models with method: {lnl_method.upper()}")
            logger.info(f"LNL Object | Predicting noisy samples by training LNL models with method: {lnl_method.upper()}")
        self._reset_lnl_models()

        # ----------- vanilla methods

        if lnl_method == 'knn':
            from pycls.lnl.KNN import KNN
            knn = KNN(self.train_data, l_set, u_set, self.cfg)
            curr_l_set_is_clean = knn.l_set_is_clean

        elif lnl_method == 'cv':
            from pycls.lnl.CrossValidation import CrossValidation
            cv = CrossValidation(self.train_data, l_set, u_set, self.cfg)
            curr_l_set_is_clean = cv.l_set_is_clean
            scores = cv.scores
            confidence_scores = cv.confidence_scores

        elif lnl_method == 'centroids':
            from pycls.lnl.centroids import Centroids
            centroids = Centroids(self.train_data, l_set, u_set, self.cfg)
            curr_l_set_is_clean = centroids.l_set_is_clean

        # ----------- from papers - Supervised Learning

        elif lnl_method == 'aum':
            from pycls.lnl.AUM import AUM
            aum = AUM(self.train_data, l_set, u_set, self.cfg)
            curr_l_set_is_clean = aum.l_set_is_clean
            scores = aum.scores
            confidence_scores = aum.confidence_scores
            self.lnl_models = aum.models

        elif lnl_method in ['map', 'disagreenet']:
            from pycls.lnl.MAP import MAP
            map_ = MAP(self.train_data, l_set, u_set, self.cfg, filtering_criterion=lnl_method)
            curr_l_set_is_clean = map_.l_set_is_clean
            scores = map_.scores
            model = EnsembleNet(map_.models)
            train_loss = map_.final_train_loss
            self.all_predictions = map_.all_train_predictions

        elif lnl_method == 'fine':
            from pycls.lnl.FINE import FINE
            fine = FINE(self.train_data, l_set, u_set, self.cfg)
            curr_l_set_is_clean = fine.l_set_is_clean

        elif lnl_method == 'imcod':
            from pycls.lnl.iMCOD import iMCOD
            imcod = iMCOD(self.train_data, l_set, u_set, self.cfg)
            curr_l_set_is_clean = imcod.l_set_is_clean

        else:
            raise NotImplementedError

        # update noise records
        lam = self.cfg.NOISE.MOMENTUM_COEFFICIENT
        curr_l_set_is_clean = np.asarray(curr_l_set_is_clean).astype(int)
        for i, pred in zip(l_set, curr_l_set_is_clean):
            if i not in self.noise_records:
                self.noise_records[i] = 0
                self.num_predictions[i] = 0
            self.num_predictions[i] += 1
            curr_prediction = 2 * pred - 1
            n = self.num_predictions[i]
            mom = lam * ((n - 1) / n)
            self.noise_records[i] = (1 - mom) * curr_prediction + mom * self.noise_records[i]

        l_set_is_clean = np.array([self.noise_records[i] >= 0 for i in l_set])

        if return_scores:
            return l_set_is_clean, scores, confidence_scores, model, train_loss
        return l_set_is_clean, model, train_loss

    def get_l_set_is_clean(self, l_set):
        return np.array([self.noise_records[i] >= 0 for i in l_set])

    def save_checkpoint(self, episode_dir, l_set=None):
        with open(f'{episode_dir}/noise_records.pkl', 'wb') as pickle_file:
            pickle.dump(self.noise_records, pickle_file)
        with open(f'{episode_dir}/num_predictions.pkl', 'wb') as pickle_file:
            pickle.dump(self.num_predictions, pickle_file)
        with open(f'{episode_dir}/lnl_models.pkl', 'wb') as pickle_file:
            pickle.dump(self.lnl_models, pickle_file)
        with open(f'{episode_dir}/all_predictions.pkl', 'wb') as pickle_file:
            pickle.dump(self.all_predictions, pickle_file)
        if l_set is not None and len(self.noise_records.keys()) == len(l_set):
            l_set_clean_is_clean = np.array([self.noise_records[i] >= 0 for i in l_set])
            np.save(f'{episode_dir}/l_set_clean_is_clean.npy', l_set_clean_is_clean)
        print(f"LNL Object | Checkpoint saved at {episode_dir}")
        logger.info(f"LNL Object | Checkpoint saved at {episode_dir}")

    def load_checkpoint(self, episode_dir):
        with open(f'{episode_dir}/noise_records.pkl', 'rb') as pickle_file:
            self.noise_records = pickle.load(pickle_file)
        with open(f'{episode_dir}/num_predictions.pkl', 'rb') as pickle_file:
            self.num_predictions = pickle.load(pickle_file)
        with open(f'{episode_dir}/lnl_models.pkl', 'rb') as pickle_file:
            self.lnl_models = pickle.load(pickle_file)
        with open(f'{episode_dir}/all_predictions.pkl', 'rb') as pickle_file:
            self.all_predictions = pickle.load(pickle_file)
        print(f"LNL Object | Checkpoint loaded from {episode_dir}")
        logger.info(f"LNL Object | Checkpoint loaded from {episode_dir}")

    @staticmethod
    def calc_noise_metrics(l_set_predicted_is_clean, l_set_true_is_clean):
        clean_tp = np.sum(np.logical_and(l_set_true_is_clean == 1, l_set_predicted_is_clean == 1))
        clean_fp = np.sum(np.logical_and(l_set_true_is_clean == 0, l_set_predicted_is_clean == 1))
        clean_tn = np.sum(np.logical_and(l_set_true_is_clean == 0, l_set_predicted_is_clean == 0))
        clean_fn = np.sum(np.logical_and(l_set_true_is_clean == 1, l_set_predicted_is_clean == 0))

        noise_accuracy = (clean_tp + clean_tn) / (clean_tp + clean_fp + clean_tn + clean_fn)

        clean_precision = clean_tp / (clean_tp + clean_fp)
        clean_recall = clean_tp / (clean_tp + clean_fn)
        clean_f1 = 2 * (clean_precision * clean_recall) / (clean_precision + clean_recall)

        noise_precision = clean_tn / (clean_tn + clean_fn)
        noise_recall = clean_tn / (clean_tn + clean_fp)
        noise_f1 = 2 * (noise_precision * noise_recall) / (noise_precision + noise_recall)

        predicted_noise = (clean_tn + clean_fn) / (clean_tn + clean_fn + clean_tp + clean_fp)
        true_noise = (clean_tn + clean_fp) / (clean_tn + clean_fn + clean_tp + clean_fp)

        return noise_accuracy, clean_precision, clean_recall, clean_f1, noise_precision, noise_recall, noise_f1, true_noise, predicted_noise
