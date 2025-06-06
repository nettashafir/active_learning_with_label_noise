import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from collections import Counter


def labels_to_unique_labels(labels, label_map):
    """
    Convert original labels to a set of unique numeric labels based on the label_map.

    :param labels: Original labels array
    :param label_map: Dictionary that maps original labels to unique integers
    :return: Array of numeric labels
    """
    return np.array([label_map[label] for label in labels])


class KNN:
    def __init__(self,
                 train_data,
                 l_set,
                 u_set,
                 cfg,

                 scale_features=False,
                 max_k=np.inf):
        """
        Initialize the KNNConsistencyCheck with the provided dataset and configuration.

        :param train_data: Dataset object containing features, noisy_labels, and targets.
        :param l_set: Indices of the labeled dataset.
        :param u_set: Indices of the unlabeled dataset (currently not used).
        :param cfg: Configuration object (can hold additional parameters).
        :param scale_features: Boolean indicating whether to scale the features.
        :param k: Number of nearest neighbors for KNN (default is 5).
        """
        # Extract features from train_data
        features = train_data.features

        # Apply scaling if specified
        if scale_features:
            print(f"KNN | Scaling the features")
            scaler = StandardScaler()
            features = scaler.fit_transform(train_data.features)

        # Select the labeled set of features and noisy labels
        self.representations = np.take(features, l_set.astype(int), axis=0)
        self.noisy_labels = np.take(train_data.noisy_labels, l_set.astype(int))
        self.true_labels = np.take(train_data.targets, l_set.astype(int))  # for debugging only

        # Map noisy labels and true labels to unique numeric values
        labels_map = {label: i for i, label in enumerate(np.unique(self.noisy_labels))}
        missing = set(self.true_labels) - set(labels_map.keys())
        for label in missing:
            labels_map[label] = len(labels_map)

        self.noisy_labels = labels_to_unique_labels(self.noisy_labels, labels_map)
        self.true_labels = labels_to_unique_labels(self.true_labels, labels_map)

        # KNN parameters
        self.l_set_is_clean = np.full(len(l_set), True)

        if len(l_set) // cfg.MODEL.NUM_CLASSES < 3:
            print(f"KNN | Number of labeled samples per class is less than 3.")

        else:
            self.k = min(max_k, len(l_set) // cfg.MODEL.NUM_CLASSES)
            # self.k = len(l_set) // cfg.MODEL.NUM_CLASSES
            self.knn_model = KNeighborsClassifier(n_neighbors=self.k)

            # Compute the class centroids and fit KNN
            print(f"KNN | Fit the k-NN model. Number of neighbors: {self.k}")
            self.fit()

            # Predict and flag noisy samples
            print(f"KNN | Predicting noisy samples")
            noisy_samples = self.predict_noisy_samples()
            self.l_set_is_clean = np.full(len(l_set), True)
            self.l_set_is_clean[noisy_samples] = False

    def fit(self):
        """
        Fit the KNN classifier on the labeled data.
        """
        self.knn_model.fit(self.representations, self.noisy_labels)

    def predict_noisy_samples(self):
        """
        Identify noisy samples based on KNN consistency.

        A sample is flagged as noisy if the majority of its k-nearest neighbors have a different label.

        :return: A list of indices of the noisy samples.
        """
        noisy_samples = []

        # Find the k-nearest neighbors for each sample and their corresponding labels
        neighbors = self.knn_model.kneighbors(self.representations, return_distance=False)

        for idx, neighbor_indices in enumerate(neighbors):
            # Get the labels of the k-nearest neighbors
            neighbor_labels = self.noisy_labels[neighbor_indices]

            # Count the occurrences of each label among the neighbors
            label_counts = Counter(neighbor_labels)

            # Find the most common label among the neighbors
            most_common_label, count = label_counts.most_common(1)[0]

            # If the sample's label is different from the most common neighbor label, mark it as noisy
            if self.noisy_labels[idx] != most_common_label:
                noisy_samples.append(idx)

        return noisy_samples
