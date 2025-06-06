import numpy as np
import torch
import torch.nn as nn
from scipy.stats import mode, beta
from sklearn.cluster import KMeans
from torch.utils.data.sampler import SubsetRandomSampler
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

from pycls.datasets.sampler import IndexedSequentialSampler
from pycls.datasets.data import MultiEpochsDataLoader


def get_indexes_data_loader(indices, batch_size, data, num_workers=4):
    assert isinstance(indices, np.ndarray), "Indexes has dtype: {} whereas expected is nd.array.".format(type(indices))
    assert isinstance(batch_size, int), "Batchsize is expected to be of int type whereas currently it has dtype: {}".format(
        type(batch_size))
    while len(indices) < batch_size:
        orig_indexes = indices
        indices = np.concatenate((indices, orig_indexes))
    subsetSampler = SubsetRandomSampler(indices)
    batch_size = min(batch_size, len(indices))
    loader = MultiEpochsDataLoader(dataset=data, batch_size=batch_size, num_workers=num_workers,
                                   sampler=subsetSampler, pin_memory=True, drop_last=True)
    return loader


def get_sequential_data_loader(indices, batch_size, data, num_workers=4):
    assert isinstance(indices, np.ndarray), "Indexes has dtype: {} whereas expected is nd.array.".format(type(indices))
    assert isinstance(batch_size, int), "Batch size is expected to be of int type whereas currently it has dtype: {}".format(type(batch_size))
    subsetSampler = IndexedSequentialSampler(indices)
    loader = MultiEpochsDataLoader(dataset=data, batch_size=batch_size, num_workers=num_workers,
                                   sampler=subsetSampler, shuffle=False, pin_memory=True)
    return loader


class EnsembleNet(nn.Module):
    def __init__(self, models, majority_vote=True):
        """
        Initializes the EnsembleNet with a list of models.

        Parameters:
        models (list of nn.Module): A list of models that are part of the ensemble.
        """
        super(EnsembleNet, self).__init__()
        models = self.flatten(models)
        self.models = nn.ModuleList(models)
        self.majority_vote = majority_vote

    @staticmethod
    def flatten(lst):
        flat_list = []
        for item in lst:
            if isinstance(item, list):
                flat_list.extend(item)
            else:
                flat_list.append(item)
        return flat_list

    def forward(self, x):
        """
        Forward pass to make predictions by majority vote from all models.

        Parameters:
        x (torch.Tensor): The input data, shape (batch_size, input_size)

        Returns:
        torch.Tensor: The predicted class labels by majority vote, shape (batch_size,)
        """
        # Collect predictions from all models
        if self.majority_vote:
            model_predictions = []

            for model in self.models:
                logits = model(x)  # Forward pass for each model
                preds = torch.argmax(logits, dim=1)  # Get the class with max logit (argmax)
                model_predictions.append(preds.unsqueeze(1))  # Append predictions with extra dimension

            # Stack predictions into a tensor (batch_size, num_models)
            all_predictions = torch.cat(model_predictions, dim=1)

            # Majority vote: Compute the mode along the model dimension (axis 1)
            majority_votes, _ = mode(all_predictions.cpu().numpy(), axis=1)

            # Convert majority vote result to a tensor
            majority_votes = torch.tensor(majority_votes).to(dtype=torch.int64, device=x.device)

            predictions = majority_votes.squeeze()  # Return the majority voted class labels

        else:
            logits = None
            for model in self.models:
                if logits is None:
                    logits = model(x)
                else:
                    logits += model(x)
            predictions = torch.argmax(logits, dim=1).to(dtype=torch.int64)

        return predictions

    def elp(self, x, y):

        model_predictions = []

        for model in self.models:
            logits = model(x)  # Forward pass for each model
            preds = torch.argmax(logits, dim=1)  # Get the class with max logit (argmax)
            model_predictions.append(preds.unsqueeze(1))  # Append predictions with extra dimension

        # Stack predictions into a tensor (batch_size, num_models)
        all_predictions = torch.cat(model_predictions, dim=1)
        predict_the_same = (all_predictions == y.reshape(-1, 1)).to(torch.float)
        agreement = predict_the_same.mean(dim=1).squeeze()
        return agreement


class BetaMixture1D(object):
    def __init__(self,
                 max_iters=50,
                 epsilon=1e-5,
                 n_components=2
                 ):
        """
        Implementation of a 1D Beta Mixture Model using EM algorithm.

        This implementation is based on the officiall code of Aranzo et al. (2019) https://arxiv.org/abs/1904.11238
        Args:
            max_iters: Maximum number of iterations for EM
            epsilon: Early stopping tolerance for th EM
            n_components: Number of components in the mixture
        """
        self.max_iters = max_iters
        self.epsilon = epsilon
        self.n_components = n_components
        self.alphas = np.zeros(n_components, dtype=np.float64)
        self.betas = np.zeros(n_components, dtype=np.float64)
        self.weight = np.zeros(n_components, dtype=np.float64)
        self.eps_nan = 1e-12

    @staticmethod
    def weighted_mean(x, w):
        return np.sum(w * x) / np.sum(w)

    def fit_beta_weighted(self, x, w):
        x_bar = self.weighted_mean(x, w)
        s2 = self.weighted_mean((x - x_bar) ** 2, w)
        alpha_ = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
        beta_ = alpha_ * (1 - x_bar) / x_bar
        return alpha_, beta_

    def likelihood(self, x, y):
        return beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(self.n_components))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r = np.array([self.weighted_likelihood(x, i) for i in range(self.n_components)])
        r[r <= self.eps_nan] = self.eps_nan  # Avoid division by zero
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def kmeans_init(self, x):
        # Initialize using KMeans clustering
        x = x.reshape(-1, 1)
        kmeans = KMeans(n_clusters=self.n_components, random_state=42).fit(x)
        labels = kmeans.labels_

        # Calculate initial parameters based on the clusters
        for i in range(self.n_components):
            cluster_data = x[labels == i].ravel()
            mean = np.mean(cluster_data)
            var = np.var(cluster_data)
            alpha = mean * ((mean * (1 - mean)) / var - 1)
            beta = alpha * (1 - mean) / mean
            self.alphas[i] = alpha
            self.betas[i] = beta
            self.weight[i] = len(cluster_data) / len(x)

    def fit(self, x, plot=False):
        x = np.copy(x)

        # EM on beta distributions is unstable with x == 0 or 1, add small eps
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        # Initialize parameters with KMeans
        self.kmeans_init(x)
        for i in range(self.max_iters):
            old_alphas = np.copy(self.alphas)
            old_betas = np.copy(self.betas)
            old_weights = np.copy(self.weight)

            # E-step
            r = self.responsibilities(x)

            # M-step
            for j in range(self.n_components):
                self.alphas[j], self.betas[j] = self.fit_beta_weighted(x, r[j])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

            # Early stopping: check if parameters have converged
            max_alpha_change = np.max(np.abs(self.alphas - old_alphas))
            max_beta_change = np.max(np.abs(self.betas - old_betas))
            max_weight_change = np.max(np.abs(self.weight - old_weights))

            if max_alpha_change < self.epsilon and max_beta_change < self.epsilon and max_weight_change < self.epsilon:
                print(f"Converged after {i + 1} iterations")
                break
        if plot:
            self.plot(x)
        return self

    def predict(self, x):
        if self.n_components == 2:
            prediction = self.posterior(x, 1) > 0.5
            mean_0 = self.alphas[0] / (self.alphas[0] + self.betas[0])
            mean_1 = self.alphas[1] / (self.alphas[1] + self.betas[1])
            if mean_0 > mean_1:
                prediction = ~prediction
            return prediction.astype(int)
        else:
            posteriors = np.array([self.posterior(x, i) for i in range(self.n_components)])
            component_assignments = np.argmax(posteriors, axis=0)
            assert component_assignments.shape == x.shape, f"Shape mismatch: {component_assignments.shape} != {x.shape}"
            return component_assignments

    def create_lookup(self, y):
        x_l = np.linspace(0 + self.eps_nan, 1 - self.eps_nan, 100)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t

    def plot(self, data):
        plt.figure(figsize=(10, 6))
        sns.histplot(data, bins=50, kde=False, color='lightgray', label='Data', stat='density')
        x_plot = np.linspace(0, 1, 100)
        plt.plot(x_plot, self.weight[0] * beta.pdf(x_plot, self.alphas[0], self.betas[0]), label='Estimated Beta 1', lw=2)
        plt.plot(x_plot, self.weight[1] * beta.pdf(x_plot, self.alphas[1], self.betas[1]), label='Estimated Beta 2', lw=2)
        # plt.plot(x_plot, self.probability(x_plot), label='Estimated Mixture', lw=2)
        plt.legend()
        plt.title('Beta Mixture Model Fitting')
        plt.xlabel('x')
        plt.ylabel('Density')
        plt.show()

    def __str__(self):
        return f'BetaMixture1D(w={self.weight}, a={self.alphas}, b={self.betas})'


def find_nearest_neighbors_set(features, l_set):
    cuda_features = torch.tensor(features).cuda()
    distances = torch.cdist(cuda_features[l_set], cuda_features)
    distances[range(len(l_set)), l_set] = np.inf
    neighbors = torch.argmin(distances, dim=1).cpu().numpy()
    neighbors = np.setdiff1d(np.unique(neighbors), l_set)
    return neighbors
