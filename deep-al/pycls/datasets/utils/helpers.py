import os
import random
import csv
import numpy as np
from torch.utils.data import Subset


def load_imageset(path, set_name):
    """
    Returns the image set `set_name` present at `path` as a list.
    Keyword arguments:
        path -- path to data folder
        set_name -- image set name - labeled or unlabeled.
    """
    reader = csv.reader(open(os.path.join(path, set_name+'.csv'), 'rt'))
    reader = [r[0] for r in reader]
    return reader

def create_subset(dataset, indices, cfg):
    noisy_labels = np.asarray(dataset.noisy_labels)[indices]
    targets = np.asarray(dataset.targets)[indices]
    is_noisy = np.asarray(dataset.is_noisy)[indices]
    noise_rate = cfg.NOISE.NOISE_RATE
    features = dataset.features[indices]
    classes = dataset.classes

    train_data = Subset(dataset, indices)
    train_size = len(train_data)

    # Update the dataset with the new subset
    train_data.targets = targets
    train_data.noisy_labels = noisy_labels
    train_data.is_noisy = is_noisy
    train_data.noise_rate = noise_rate
    train_data.features = features
    train_data.classes = classes

    return train_data, train_size


def add_asym_label_noise(labels, transition_matrix, noise_rate):
    """
    Add noise to classification labels based on a transition matrix (vectorized implementation).
    The number of corrupted samples per class is proportional to (1 - T[i,i]/n_classes).

    Parameters:
    -----------
    labels : array-like
        Original classification labels (integers starting from 0)
    transition_matrix : ndarray
        Square matrix where entry (i,j) represents probability of transitioning
        from class i to class j
    noise_rate : float
        Proportion of labels to be corrupted (between 0 and 1)

    Returns:
    --------
    noisy_labels : ndarray
        Labels with added noise
    noise_mask : ndarray
        Boolean mask indicating which labels were corrupted
    """
    labels = np.asarray(labels)
    n_samples = len(labels)
    n_classes = transition_matrix.shape[0]

    # Validate inputs
    if transition_matrix.shape[0] != transition_matrix.shape[1]:
        raise ValueError("Transition matrix must be square")
    if not 0 <= noise_rate <= 1:
        raise ValueError("Noise rate must be between 0 and 1")
    if labels.max() >= n_classes:
        raise ValueError("Labels contain classes not covered by transition matrix")

    # Create copy of labels
    noisy_labels = labels.copy()

    # Calculate corruption ratios for each class based on transition matrix
    diag_probs = np.diag(transition_matrix)
    corruption_ratios = 1 - diag_probs / n_classes
    corruption_ratios = corruption_ratios / corruption_ratios.sum()  # Normalize to sum to 1

    # Calculate number of samples to corrupt per class
    n_corrupt_total = int(n_samples * noise_rate)
    n_corrupt_per_class = np.round(corruption_ratios * n_corrupt_total).astype(int)

    # Adjust for rounding errors to match total desired corruptions
    n_corrupt_per_class[-1] = n_corrupt_total - n_corrupt_per_class[:-1].sum()

    # Initialize noise mask
    noise_mask = np.zeros(n_samples, dtype=bool)
    corrupt_idx = []

    # Sample indices for each class
    for class_idx in range(n_classes):
        # Get indices of samples in this class
        class_samples = np.where(labels == class_idx)[0]

        # Skip if no samples to corrupt for this class
        if n_corrupt_per_class[class_idx] <= 0 or len(class_samples) == 0:
            continue

        # Adjust if trying to corrupt more samples than available
        n_corrupt = min(n_corrupt_per_class[class_idx], len(class_samples))

        # Sample indices from this class
        sampled_idx = np.random.choice(
            class_samples,
            size=n_corrupt,
            replace=False
        )
        corrupt_idx.extend(sampled_idx)

    corrupt_idx = np.array(corrupt_idx)
    noise_mask[corrupt_idx] = True

    if len(corrupt_idx) > 0:
        # Get transition probabilities for corrupted labels
        corrupt_labels = labels[corrupt_idx]
        trans_probs = transition_matrix[corrupt_labels]

        # Set self-transition probabilities to 0
        trans_probs[np.arange(len(corrupt_idx)), corrupt_labels] = 0

        # Normalize remaining probabilities
        row_sums = trans_probs.sum(axis=1, keepdims=True)
        trans_probs = trans_probs / row_sums

        # Sample new labels for all corrupted indices at once
        cumsum_probs = np.cumsum(trans_probs, axis=1)
        random_values = np.random.random(len(corrupt_idx))[:, np.newaxis]
        new_labels = (random_values < cumsum_probs).argmax(axis=1)

        # Update corrupted labels
        noisy_labels[corrupt_idx] = new_labels

    return noisy_labels
