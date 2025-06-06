import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
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


class CrossValidation:
    def __init__(self,
                 train_data,
                 l_set,
                 u_set,
                 cfg,

                 scale_features=False,
                 k_folds=3):
        """
        Initialize CrossValidationNoiseDetection with the provided dataset and configuration.

        :param train_data: Dataset object containing features, noisy_labels, and targets.
        :param l_set: Indices of the labeled dataset.
        :param u_set: Indices of the unlabeled dataset (currently not used).
        :param cfg: Configuration object (can hold additional parameters).
        :param scale_features: Boolean indicating whether to scale the features.
        :param k_folds: Number of cross-validation folds (default is 5).
        """
        # Extract features from train_data
        features = train_data.features

        # Apply scaling if specified
        if scale_features:
            print(f"Cross-Validation | Scaling the features")
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

        # K-Fold cross-validation parameters
        # self.k_folds = max(min(max_k_folds, len(l_set) // cfg.MODEL.NUM_CLASSES), 2)
        self.k_folds = k_folds
        self.l_set = l_set
        self.cfg = cfg

        # Detect noisy samples using cross-validation
        print(f"Cross-Validation | Detecting noisy samples with {self.k_folds}-fold cross-validation")
        l_set_is_clean, scores, confidence_scores = self.detect_noisy_samples_via_cv()

        self.l_set_is_clean = l_set_is_clean
        self.confidence_scores = confidence_scores
        self.scores = scores

    def _calculate_confidence_scores(self, disagree_count, model_predictions, model_confidences, l_set_is_clean, n_samples):
        """
        For noise detection based on cross-validation with k models, we compute confidence scores by combining two factors:

            1. The agreement ratio - the proportion of models agreeing with the majority decision (ranging from just over 50% to 100%)
            2. The average confidence (probability scores) of the agreeing models

        The final confidence score is the product of these factors.
        This results in high confidence when most models agree and predict their labels with high probability, and low
        confidence when there is either weak agreement between models or the agreeing models have low confidence in their predictions.
        Alternative phrasing for a more technical audience:
        For each sample, the confidence score is calculated by multiplying two factors:

            1. the ratio of models agreeing with the majority decision, and
            2. the mean probability scores of the agreeing models' predictions.

        This captures both inter-model consensus and intra-model confidence, providing high scores when most models
        strongly agree on a prediction and low scores when there is either weak agreement between models or low
        prediction confidence among the agreeing models.
        """
        # Agreement factor: what portion of models agree with majority
        agreement_ratios = np.maximum(disagree_count, self.k_folds - disagree_count) / self.k_folds


        # Create masks for predictions matching/not matching noisy labels
        predictions_match = model_predictions == self.noisy_labels
        predictions_differ = ~predictions_match

        # Get confidences for agreeing predictions
        clean_mask = l_set_is_clean
        noisy_mask = ~l_set_is_clean

        # Use masked array to handle selection of relevant confidences
        masked_conf_clean = np.ma.array(model_confidences, mask=~predictions_match)
        masked_conf_noisy = np.ma.array(model_confidences, mask=~predictions_differ)

        majority_confidences = np.zeros(n_samples)
        majority_confidences[clean_mask] = np.ma.mean(masked_conf_clean[:, clean_mask], axis=0)
        majority_confidences[noisy_mask] = np.ma.mean(masked_conf_noisy[:, noisy_mask], axis=0)

        # # Equivalent to:
        # majority_confidences2 = []
        # for i in range(n_samples):
        #     if not l_set_is_clean[i]:
        #         # For noisy samples, take average confidence of models that predicted different from given label
        #         conf = model_confidences[model_predictions[:, i] != self.noisy_labels[i]][:, i].mean()
        #     else:
        #         # For clean samples, take average confidence of models that predicted same as given label
        #         conf = model_confidences[model_predictions[:, i] == self.noisy_labels[i]][:, i].mean()
        #     majority_confidences2.append(conf)
        # majority_confidences2 = np.array(majority_confidences)

        # Combine both factors
        confidence_scores = agreement_ratios * majority_confidences

        return confidence_scores

    def _extract_given_label_probabilities(self, probability_scores, trained_classes):
        num_classes = len(np.unique(self.noisy_labels))
        num_samples = len(self.noisy_labels)
        class_to_index = np.full(num_classes, -1)  # Initialize with -1 (invalid index)
        class_to_index[trained_classes] = np.arange(len(trained_classes))

        # Map y_test to indices in the model's probability array
        mapped_indices = class_to_index[self.noisy_labels]

        # Extract probabilities for the test set labels
        # For missing classes (index -1), assign probability 0
        given_label_probs = np.where(mapped_indices != -1, probability_scores[np.arange(num_samples), mapped_indices], 0.0)

        class_to_index = {cls: idx for idx, cls in enumerate(trained_classes)}

        # Extract the probabilities for the given labels in y_test
        given_label_probs2 = np.array([
            probability_scores[i, class_to_index[label]] if label in class_to_index else 0.0
            for i, label in enumerate(self.noisy_labels)
        ])

        return given_label_probs

    def detect_noisy_samples_via_cv(self):
        """
        Detect noisy samples using K-fold cross-validation with SVM.

        A sample is flagged as noisy if the model consistently predicts a different label for it in cross-validation.

        :return: A list of indices of the noisy samples.
        """
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.cfg.RNG_SEED)
        n_samples = len(self.l_set)
        disagree_count = np.zeros(n_samples, dtype=int)
        model_predictions = np.zeros((self.k_folds, n_samples), dtype=int)
        model_confidences = np.zeros((self.k_folds, n_samples), dtype=float)
        scores = np.zeros(n_samples, dtype=float)
        # model_confidences_in_given_label = np.zeros((self.k_folds, n_samples), dtype=float)

        # Iterate over each fold
        for i, (val_index, train_index) in enumerate(kf.split(self.representations)):
            X_train, X_val = self.representations[train_index], self.representations[val_index]
            y_train, y_val = self.noisy_labels[train_index], self.noisy_labels[val_index]

            # Train an SVM model on the current training split
            linear_model = LogisticRegression(random_state=self.cfg.RNG_SEED)
            linear_model.fit(X_train, y_train)

            # Predict on the validation set
            y_pred = linear_model.predict(self.representations)
            probability_scores = linear_model.predict_proba(self.representations)

            # Compare predictions with noisy labels, count misclassifications
            disagree_count += (y_pred != self.noisy_labels)
            model_predictions[i] = y_pred
            model_confidences[i] = np.max(probability_scores, axis=1)
            scores += self._extract_given_label_probabilities(probability_scores, linear_model.classes_)

        # A sample is flagged as noisy if it was misclassified in most folds
        threshold = self.k_folds / 2  # Flag as noisy if misclassified in more than half the folds
        l_set_is_clean = ~(disagree_count > threshold)

        # Combine both factors
        print(f"Cross-Validation | Calculating confidence scores")
        confidence_scores = self._calculate_confidence_scores(disagree_count, model_predictions, model_confidences, l_set_is_clean, n_samples)

        return l_set_is_clean, scores, confidence_scores


# Example usage
if __name__ == "__main__":
    # Assuming train_data has 'features', 'noisy_labels', and 'targets'
    # l_set and u_set are arrays of indices for labeled and unlabeled samples

    train_data = ...  # Dataset object with required attributes
    l_set = np.array([0, 1, 2, 3, 4, 5, 6])  # Example labeled set indices
    u_set = np.array([7, 8, 9])  # Example unlabeled set indices (unused here)
    cfg = {}  # Example configuration dictionary

    # Initialize the CrossValidationNoiseDetection
    cv_noise_detector = CrossValidation(train_data, l_set, u_set, cfg, scale_features=True, max_k_folds=5)

    # Access the result
    print("Clean samples:", cv_noise_detector.l_set_is_clean)
