import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


def labels_to_unique_labels(labels, label_map):
    """
    Convert original labels to a set of unique numeric labels based on the label_map.

    :param labels: Original labels array
    :param label_map: Dictionary that maps original labels to unique integers
    :return: Array of numeric labels
    """
    return np.array([label_map[label] for label in labels])


class DBSCANNoiseDetector:
    def __init__(self,
                 train_data,
                 l_set,
                 u_set,
                 cfg,
                 scale_features=False):
        """
        Initialize DBSCANNoiseDetector with the provided dataset and configuration.

        :param train_data: Dataset object containing features, noisy_labels, and targets.
        :param l_set: Indices of the labeled dataset.
        :param u_set: Indices of the unlabeled dataset (currently not used).
        :param cfg: Configuration object (can hold additional parameters).
        :param scale_features: Boolean indicating whether to scale the features.
        """
        # Extract features from train_data
        features = train_data.features

        # Apply scaling if specified
        if scale_features:
            print(f"DBSCAN Noise Detector | Scaling the features")
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

        # Compute the covariance matrix of the features
        if self.representations.shape[0] < self.representations.shape[1]:
            covariance_matrix = np.cov(self.representations)
        else:
            covariance_matrix = np.cov(self.representations, rowvar=False)

        # Compute the trace of the covariance matrix
        self.trace_cov = np.trace(covariance_matrix)
        self.d = covariance_matrix.shape[1]

        # Compute covariance matrix and augment features
        print(f"DBSCAN Noise Detector | Computing covariance matrix and augmenting features")
        augmented_features = self.augment_features_with_labels()

        # Apply DBSCAN clustering and detect noisy samples
        print(f"DBSCAN Noise Detector | Running DBSCAN clustering")
        noisy_samples = self.detect_noisy_samples(augmented_features)

        self.l_set_is_clean = np.full(len(l_set), True)
        self.l_set_is_clean[noisy_samples] = False

    def augment_features_with_labels(self):
        """
        Compute the covariance matrix of the features, and augment the features by adding the label
        as an additional feature, weighted by the mean variance of the features.

        :return: Augmented features.
        """
        mean_variance = self.trace_cov / self.d

        # Augment the features by adding the label as an additional feature, weighted by mean_variance
        labels_as_feature = self.noisy_labels[:, np.newaxis] * mean_variance * 10
        augmented_features = np.hstack([self.representations, labels_as_feature])

        return augmented_features

    def detect_noisy_samples(self, augmented_features):
        """
        Run DBSCAN clustering on the augmented features, with epsilon based on the determinant of the covariance matrix.

        :param augmented_features: Features augmented with labels as an additional dimension.
        :return: Indices of noisy samples (outliers detected by DBSCAN).
        """
        # # Compute the covariance matrix of the original features
        # if self.representations.shape[0] < self.representations.shape[1]:
        #     covariance_matrix = np.cov(self.representations)
        # else:
        #     covariance_matrix = np.cov(self.representations, rowvar=False)
        #
        # # Compute epsilon as the 1/d-th root of the determinant of the covariance matrix
        # d = covariance_matrix.shape[1]
        # det_cov = np.linalg.det(covariance_matrix)
        # epsilon = det_cov ** (1 / d)
        epsilon = self.trace_cov  # / self.d

        # Run DBSCAN clustering on the augmented features
        dbscan = DBSCAN(eps=epsilon)
        dbscan_labels = dbscan.fit_predict(augmented_features)

        # Samples labeled as -1 by DBSCAN are considered as outliers (noisy samples)
        noisy_samples = np.where(dbscan_labels == -1)[0]

        return noisy_samples


# Example usage
if __name__ == "__main__":
    # Assuming train_data has 'features', 'noisy_labels', and 'targets'
    # l_set and u_set are arrays of indices for labeled and unlabeled samples

    train_data = ...  # Dataset object with required attributes
    l_set = np.array([0, 1, 2, 3, 4, 5, 6])  # Example labeled set indices
    u_set = np.array([7, 8, 9])  # Example unlabeled set indices (unused here)
    cfg = {}  # Example configuration dictionary

    # Initialize the DBSCANNoiseDetector
    dbscan_noise_detector = DBSCANNoiseDetector(train_data, l_set, u_set, cfg, scale_features=True)

    # Access the result
    print("Clean samples:", dbscan_noise_detector.l_set_is_clean)
