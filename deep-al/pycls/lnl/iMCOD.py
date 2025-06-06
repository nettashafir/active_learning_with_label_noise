import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def labels_to_unique_labels(labels, label_map):
    """
    Convert original labels to a set of unique numeric labels based on the label_map.
    """
    return np.array([label_map[label] for label in labels])


class iMCOD:
    def __init__(self,
                 train_data,
                 l_set,
                 u_set,
                 cfg,
                 scale_features=False,
                 max_k=np.inf,
                 threshold=1e-4,
                 kernel='rbf',
                 C=1.0):
        """
        Initialize the iMCOD model with the labeled and unlabeled datasets.

        :param train_data: Dataset object containing features, noisy_labels, and targets.
        :param l_set: Indices of the labeled dataset.
        :param u_set: Indices of the unlabeled dataset (not used in this method).
        :param cfg: Configuration object with parameters (e.g., number of classes).
        :param scale_features: Boolean to scale features.
        :param max_k: Maximum number of nearest neighbors for iMCOD (default is infinity).
        :param threshold: Threshold for CB-MoNNAD outlier detection.
        :param kernel: Kernel type for SVM (default is 'rbf').
        :param C: Regularization parameter for SVM.
        """
        # Extract features
        features = train_data.features

        # Apply scaling if needed
        if scale_features:
            scaler = StandardScaler()
            features = scaler.fit_transform(features)

        # Subset the labeled data
        self.representations = np.take(features, l_set.astype(int), axis=0)
        self.noisy_labels = np.take(train_data.noisy_labels, l_set.astype(int))
        self.true_labels = np.take(train_data.targets, l_set.astype(int))  # For debugging only

        # Create label mappings
        labels_map = {label: i for i, label in enumerate(np.unique(self.noisy_labels))}
        missing = set(self.true_labels) - set(labels_map.keys())
        for label in missing:
            labels_map[label] = len(labels_map)

        self.noisy_labels = labels_to_unique_labels(self.noisy_labels, labels_map)
        self.true_labels = labels_to_unique_labels(self.true_labels, labels_map)

        # Set k for nearest neighbors
        self.k = min(max_k, len(l_set) // cfg.MODEL.NUM_CLASSES)
        self.threshold = threshold
        self.l_set_is_clean = np.full(len(l_set), True)

        # Train kNN model only once per class
        print("iMCOD | Fitting kNN models")
        self.knn_models = self.fit()

        # Predict and flag noisy samples
        print("iMCOD | Predicting noisy samples")
        noisy_samples = self.predict_noisy_samples()
        self.l_set_is_clean = np.full(len(l_set), True)
        self.l_set_is_clean[noisy_samples] = False

    def fit(self):
        """
        Train a kNN model for each class and store the models.
        """
        knn_models = {}
        unique_classes = np.unique(self.noisy_labels)

        for class_label in unique_classes:
            class_indices = np.where(self.noisy_labels == class_label)[0]
            class_representations = self.representations[class_indices]

            neighbors = NearestNeighbors(n_neighbors=min(self.k, len(class_indices)))
            neighbors.fit(class_representations)
            knn_models[class_label] = (neighbors, class_indices)

        return knn_models

    def _class_based_knn(self, query, query_class):
        """
        Find the k nearest neighbors of a query instance within the same class.
        """
        knn_class, class_indices = self.knn_models[query_class]

        # Compute distances and indices within the same class
        distances, indices = knn_class.kneighbors([query])

        # Map the indices back to the original dataset indices
        original_indices = class_indices[indices[0]]

        return distances, original_indices

    def _local_outlier_factor(self, neighbors, query_idx):
        """
        Compute LOF for a query instance using its nearest neighbors.
        Use vectorized operations for efficiency.
        """
        neighbor_vectors = self.representations[neighbors]
        query_vector = self.representations[query_idx]

        # Calculate pairwise distances between query and neighbors
        distances = np.linalg.norm(neighbor_vectors - query_vector, axis=1)
        lrd = np.mean(1 / (distances + 1e-5))

        # Calculate LOF score
        lof = np.mean(distances) / lrd
        return lof

    def _cb_monad(self, neighbors, query_idx):
        """
        Compute CB-MoNNAD score for a query instance based on its class neighbors.
        Use vectorized operations to calculate LOF values.
        """
        loff_values = np.array([self._local_outlier_factor(neighbors, n) for n in neighbors])
        cb_monnad = np.median(np.abs(loff_values - self._local_outlier_factor(neighbors, query_idx)))
        return cb_monnad

    def predict_noisy_samples(self):
        """
        Detect noisy samples based on the CB-MoNNAD score exceeding the threshold.
        Use vectorized operations where possible.
        """
        noisy_samples = []

        for idx, query in enumerate(self.representations):
            query_class = self.noisy_labels[idx]

            # Find k nearest neighbors within the same class
            distances, neighbors = self._class_based_knn(query, query_class)

            # Compute CB-MoNNAD score
            cb_monnad_score = self._cb_monad(neighbors, idx)

            # Mark sample as noisy if CB-MoNNAD score exceeds the threshold
            if cb_monnad_score > self.threshold:
                noisy_samples.append(idx)

        return noisy_samples
