import os
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

from .lnl_utils import BetaMixture1D


class FINE:
    def __init__(self,
                 train_data,
                 l_set,
                 u_set,
                 cfg,

                 scale_features=True,
                 use_bmm=False,
                 ):
        """
        Initialize the FINE algorithm for clean sample identification.

        Described in Kim et al. (2021) https://arxiv.org/abs/2102.11628
        """
        features = train_data.features
        if scale_features:
            print(f"FINE | Scaling the features")
            scaler = StandardScaler()
            features = scaler.fit_transform(train_data.features)

        self.representations = np.take(features, l_set.astype(int), axis=0)
        self.noisy_labels = np.take(train_data.noisy_labels, l_set.astype(int))
        self.true_labels = np.take(train_data.targets, l_set.astype(int))  # for debugging only
        self.n_classes = cfg.MODEL.NUM_CLASSES
        self.gram_matrices = {k: np.zeros((self.representations.shape[1], self.representations.shape[1]))
                              for k in range(self.n_classes)}
        self.eigenvectors = {}
        self.use_bmm = use_bmm

        print(f"FINE | Compute gram matrix")
        self.compute_gram_matrices()
        print(f"FINE | Compute eigenvectors")
        self.compute_eigenvectors()
        print(f"FINE | Compute fine scores and clean noisy samples")
        self.l_set_is_clean  = self.filter_clean_samples()

    def compute_gram_matrices(self):
        """Compute the gram matrix for each class based on the representations."""
        for x, y in zip(self.representations, self.noisy_labels):
            self.gram_matrices[y] += np.outer(x, x)

    def compute_eigenvectors(self):
        """Perform eigen decomposition to find the principal eigenvector for each class."""
        for k in range(self.n_classes):
            eigvals, eigvecs = np.linalg.eig(self.gram_matrices[k])
            self.eigenvectors[k] = eigvecs[:, np.argmax(eigvals)]  # Select the eigenvector with the largest eigenvalue

    def compute_fine_scores(self):
        """Compute FINE scores for each sample."""
        fine_scores = []
        for x, y in zip(self.representations, self.noisy_labels):
            principal_eigenvector = self.eigenvectors[y]
            score = np.dot(principal_eigenvector, x) ** 2  # Alignment score squared
            fine_scores.append(score)
        return np.array(fine_scores)

    def filter_clean_samples(self):
        """Use Gaussian Mixture Model (GMM) to filter clean samples based on FINE scores."""
        fine_scores_by_class = {k: [] for k in range(self.n_classes)}
        indices_by_class = {k: [] for k in range(self.n_classes)}

        # Collect FINE scores by class
        fine_scores = self.compute_fine_scores()
        for index, (score, label) in enumerate(zip(fine_scores, self.noisy_labels)):
            fine_scores_by_class[label].append(score)
            indices_by_class[label].append(index)

        # Apply GMM to separate clean and noisy samples
        l_set_is_clean = np.full(len(self.noisy_labels), False)
        for k in range(self.n_classes):

            scores = np.array(fine_scores_by_class[k])
            if len(scores) < 2:
                l_set_is_clean[indices_by_class[k]] = True
                continue

            if self.use_bmm:
                scores_max = np.max(scores)
                scores_min = np.min(scores)
                scores = (scores - scores_min) / (scores_max - scores_min)
                eps = 1e-8
                scores = np.clip(scores, eps, 1 - eps)
                bmm = BetaMixture1D()
                bmm.fit(scores)
                l_set_is_clean[indices_by_class[k]] = bmm.predict(scores)
            else:
                scores = scores.reshape(-1, 1)
                gmm = GaussianMixture(n_components=2)
                gmm.fit(scores)
                probabilities = gmm.predict_proba(scores)
                clean_cluster = np.argmax(gmm.means_)
                l_set_is_clean[[indices_by_class[k][i] for i, p in enumerate(probabilities[:, clean_cluster]) if p > 0.5]] = True

        return l_set_is_clean
