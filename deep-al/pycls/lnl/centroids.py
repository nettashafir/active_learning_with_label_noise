import numpy as np
from sklearn.preprocessing import StandardScaler
from numpy.linalg import det
from scipy.special import binom


def labels_to_unique_labels(labels, labels_map: dict):
    unique_labels = np.array([labels_map[label] for label in labels])
    return unique_labels


def mahalanobis_distance(x, mean, cov_matrix):
    """
    Calculate Mahalanobis distance using Cholesky decomposition for covariance matrix inversion.

    Parameters:
    - x: The data point (1D array)
    - mean: Mean vector of the distribution (1D array)
    - cov_matrix: Covariance matrix (2D array, symmetric and positive definite)

    Returns:
    - Mahalanobis distance (float)
    """
    # Compute x - mean
    x_minus_mean = x - mean

    # Cholesky decomposition of the covariance matrix (cov_matrix = L * L.T)
    L = np.linalg.cholesky(cov_matrix + 1e-10 * np.eye(cov_matrix.shape[0]))

    # Solve for the system L * y = (x - mean) using forward substitution
    y = np.linalg.solve(L, x_minus_mean)

    # Compute the Mahalanobis distance as sqrt(y.T * y)
    distance = np.sqrt(np.dot(y.T, y))

    return distance


class Centroids:
    def __init__(self,
                 train_data,
                 l_set,
                 u_set,
                 cfg,

                 scale_features=False,
                 use_mahalanobis=False,
                 min_B=15,
                 max_B=30,
                 subset_fraction=0.5,
                 min_samples_per_class=2,
                 max_samples_per_class=50):
        """
        Initialize the NoisySampleDetector.

        :param representations: A numpy array of shape (n_samples, n_features)
                                where each row is the feature representation of a sample.
        :param labels: A numpy array of shape (n_samples,) containing the class labels for each sample.
        """
        # features
        features = train_data.features
        if scale_features:
            print(f"Centroids | Scaling the features")
            scaler = StandardScaler()
            features = scaler.fit_transform(train_data.features)
        self.representations = np.take(features, l_set.astype(int), axis=0)
        # self.representations = StandardScaler().fit_transform(self.representations)

        # labels
        self.noisy_labels = np.take(train_data.noisy_labels, l_set.astype(int))
        self.true_labels = np.take(train_data.targets, l_set.astype(int))  # for debugging only
        labels_map = {label: i for i, label in enumerate(np.unique(self.noisy_labels))}
        missing = set(self.true_labels) - set(labels_map.keys())
        for label in missing:
            labels_map[label] = len(labels_map)
        self.noisy_labels = labels_to_unique_labels(self.noisy_labels, labels_map)
        self.true_labels = labels_to_unique_labels(self.true_labels, labels_map)

        # centroids configuration
        self.use_mahalanobis = use_mahalanobis
        self.class_centroids = {}
        self.class_covariances = {}
        self.min_B = min_B
        self.max_B = max_B
        self.subset_fraction = subset_fraction
        self.min_samples_per_class = min_samples_per_class
        self.max_samples_per_class = max_samples_per_class

        print(f"Centroids | Calculating class centroids")
        self.fit()
        print(f"Centroids | Predicting noisy samples")
        noisy_samples = self.predict_noisy_samples()
        self.l_set_is_clean = np.full(len(l_set), True)
        self.l_set_is_clean[noisy_samples] = False

    def fit(self):
        """
        Calculate the centroid of each class by taking multiple random subsets of points from each class,
        computing the mean and determinant of the covariance matrix for each subset, and selecting the
        mean that corresponds to the lowest covariance determinant.
        """
        unique_classes = np.unique(self.noisy_labels)

        for cls in unique_classes:
            class_representations = self.representations[(self.noisy_labels == cls)]

            n_class_samples = class_representations.shape[0]
            if n_class_samples < 3:
                # If the class has fewer than 3 samples, use the mean of all samples as the centroid
                self.class_centroids[cls] = np.mean(class_representations, axis=0)
                self.class_covariances[cls] = np.cov(class_representations, rowvar=False)
                continue

            # Number of samples to select in each subset
            subset_size = int(self.subset_fraction * n_class_samples)
            subset_size = min(max(subset_size, self.min_samples_per_class), self.max_samples_per_class)
            B = int(np.log(binom(n_class_samples, subset_size)))
            B = min(max(B, self.min_B), self.max_B)

            # # Initialize the best centroid and covariance determinant as the mean and determinant of all samples
            # best_mean = np.mean(class_representations, axis=0)
            # if class_representations.shape[0] < class_representations.shape[1]:
            #     covariance = np.cov(class_representations)
            # else:
            #     covariance = np.cov(class_representations, rowvar=False)
            # lowest_determinant = det(covariance)

            best_mean = None
            best_covariance = None
            lowest_determinant = float('inf')

            # Repeat B times
            for _ in range(B):
                # Randomly select a subset of samples
                subset_indices = np.random.choice(n_class_samples, subset_size, replace=False)
                subset = class_representations[subset_indices]

                # Compute mean of the subset
                subset_mean = np.mean(subset, axis=0)

                # Compute covariance matrix and its determinant
                if subset.shape[0] < subset.shape[1]:
                    subset_covariance = np.cov(subset)
                else:
                    subset_covariance = np.cov(subset, rowvar=False)

                subset_determinant = abs(det(subset_covariance))

                # Check if this subset has the lowest determinant so far
                if subset_determinant < lowest_determinant:
                    lowest_determinant = subset_determinant
                    best_mean = subset_mean
                    if self.use_mahalanobis:
                        best_covariance = np.cov(subset, rowvar=False)

            # Store the best centroid for the class
            self.class_centroids[cls] = best_mean
            self.class_covariances[cls] = best_covariance

    def predict_noisy_samples(self):
        """
        Identify noisy samples based on their distance to class centroids.

        A sample is flagged as noisy if it's closer to a centroid of another class than to its own class's centroid.

        :return: A list of indices of the noisy samples.
        """
        noisy_samples = []
        for idx, (sample, label) in enumerate(zip(self.representations, self.noisy_labels)):
            # Calculate distance to own class's centroid
            own_centroid = self.class_centroids[label]

            if self.use_mahalanobis:
                own_covariance = self.class_covariances[label]
                own_distance = mahalanobis_distance(sample, own_centroid, own_covariance)
            else:
                own_distance = np.linalg.norm(sample - own_centroid)

            # Calculate distance to other class centroids
            other_distances = {cls: np.linalg.norm(sample - centroid)
                               for cls, centroid in self.class_centroids.items() if cls != label}

            # Find the closest other class centroid
            closest_other_class, closest_other_distance = min(other_distances.items(), key=lambda x: x[1])

            # If the sample is closer to another class's centroid, flag it as noisy
            if closest_other_distance < own_distance:
                noisy_samples.append(idx)

        return noisy_samples


