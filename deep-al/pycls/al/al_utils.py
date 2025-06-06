import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm
from scipy.special import logsumexp


def construct_graph(features, delta, cosine_dist, batch_size=500):
    """
    creates a directed graph where:
    x->y iff l2(x,y) < delta.

    represented by a list of edges (a sparse matrix).
    stored in a dataframe
    """
    xs, ys, ds = [], [], []
    # distance computations are done in GPU
    cuda_feats = torch.tensor(features).cuda()
    for i in range(len(features) // batch_size):
        # distance comparisons are done in batches to reduce memory consumption
        cur_feats = cuda_feats[i * batch_size: (i + 1) * batch_size]
        if cosine_dist:
            dist = (1 - cosine_similarity(cur_feats, cuda_feats))
        else:
            dist = torch.cdist(cur_feats, cuda_feats)
        mask = dist < delta
        # saving edges using indices list - saves memory.
        x, y = mask.nonzero().T
        xs.append(x.cpu() + batch_size * i)
        ys.append(y.cpu())
        ds.append(dist[mask].cpu())

    xs = torch.cat(xs).numpy()
    ys = torch.cat(ys).numpy()
    ds = torch.cat(ds).numpy()

    df = pd.DataFrame({'x': xs, 'y': ys, 'd': ds})
    return df

def construct_graph_for_large_datasets(features, delta, cosine_dist, batch_size=500):
    """
    Creates a directed graph where:
    x->y iff l2(x, y) < delta.

    Represented as a list of edges in a Pandas DataFrame with columns: x, y, d.

    Parameters:
        features (numpy.ndarray): MxD matrix of dataset features.
        delta (float): Distance threshold for edge creation.
        batch_size (int): Size of feature batches for distance computation.

    Returns:
        pd.DataFrame: DataFrame containing edges with columns ['x', 'y', 'd'].
    """
    xs, ys, ds = [], [], []
    cuda_feats = torch.tensor(features, dtype=torch.float32).cuda()
    num_samples = len(features)

    for i in tqdm(range((num_samples + batch_size - 1) // batch_size)):  # Outer batch loop
        cur_feats = cuda_feats[i * batch_size: (i + 1) * batch_size]

        for j in range((num_samples + batch_size - 1) // batch_size):  # Inner batch loop
            other_feats = cuda_feats[j * batch_size: (j + 1) * batch_size]

            # Compute pairwise distances between the two batches
            if cosine_dist:
                dist = (1 - cosine_similarity(cur_feats, cuda_feats))
            else:
                dist = torch.cdist(cur_feats, cuda_feats)
            mask = dist < delta
            x, y = mask.nonzero(as_tuple=True)

            # Map indices to global dataset indices
            xs.append(x.cpu() + i * batch_size)
            ys.append(y.cpu() + j * batch_size)
            ds.append(dist[mask].cpu())

    xs = torch.cat(xs).numpy()
    ys = torch.cat(ys).numpy()
    ds = torch.cat(ds).numpy()

    df = pd.DataFrame({'x': xs, 'y': ys, 'd': ds})
    return df


def build_graph(features, delta, cosine_dist, batch_size=500):
    if len(features) <= 100_000:
        graph = construct_graph(features, delta, cosine_dist, batch_size)
    else:
        graph = construct_graph_for_large_datasets(features, delta, cosine_dist, batch_size)
    return graph


def choose_delta_for_probcover(features, classes_num, cosine_dist, alpha=0.95, df=None):
    """
    called only in the beginning when len(lSet)=0.
    @return:
    """
    # construct pseudo labels using K-means
    train_size = len(features)
    if train_size >= 500_000:
        print("Creating a subset of the training data for faster computation.")
        subset_size = int(0.1 * train_size)
        indices = np.random.choice(train_size, size=subset_size, replace=False)
        features = features[indices]
    print("Construct pseudo-labels using K-means")
    # normalized_features = features / np.linalg.norm(features, axis=1, keepdims=True)
    normalized_features = features
    k_means = KMeans(n_clusters=int(classes_num))
    k_means.fit(normalized_features)
    pseudo_labels = k_means.predict(normalized_features)

    # calculate the purity using running delta-s
    print("Bullding graph")
    if cosine_dist:
        all_del_val = np.arange(0.02, 0.32, 0.02)[::-1]
    else:
        all_del_val = np.arange(0.02, 0.62, 0.02)[::-1]
    if df is None:
        df = build_graph(normalized_features, all_del_val[0], cosine_dist)
    purity_val_lst = []
    print("Calculating purity for each delta")
    for delta in tqdm(all_del_val, desc="deltas"):
        # construct the graph
        df = df[df.d < delta]

        # for each vertex - check if it is pure (if its neighbors k-means label is equal to its)
        num_pure_vertices = 0
        grouped_df = df.groupby('x')
        for curr_x, group in grouped_df:
            vertex_label = pseudo_labels[curr_x]
            neighbors_y = group['y'].unique().tolist()
            neighbor_labels = pseudo_labels[neighbors_y] # list(map(pseudo_labels.__getitem__, neighbors_y))
            if all(neighbor_label == vertex_label for neighbor_label in neighbor_labels):
                num_pure_vertices += 1

        # compute num_of_pure_vertices/num_of_all_vertices
        purity = num_pure_vertices / len(normalized_features)
        purity_val_lst.append(purity)
        print(f"delta - {delta}, num of edges - {len(df)}, purity - {purity}")

        del grouped_df

    # choose the maximal delta that gets 0.95 purity at least
    deltas = all_del_val[np.array(purity_val_lst) >= alpha]
    res = float(max(deltas))
    print("Optional deltas: ", all_del_val)
    print("Purities: ", purity_val_lst)
    print("The chosen delta: ", res)
    return res

def calculate_coverage(features, l_set, u_set, l_set_is_clean, delta, cosine_dist=True):
    print(f"Calculating coverage using delta {delta}")
    graph = build_graph(features, delta, cosine_dist)
    edge_from_seen = np.isin(graph.x, l_set[l_set_is_clean])
    covered_samples = graph.y[edge_from_seen].unique()
    coverage = (len(covered_samples) + len(l_set[~l_set_is_clean])) / (len(l_set) + len(u_set))
    return coverage

def shannon_entropy(distribution):
    """Calculates the Shannon entropy of a probability distribution.

    Args:
        distribution: A numpy array representing the probability distribution.

    Returns:
        The Shannon entropy of the distribution.
    """
    distribution = np.asarray(distribution)
    distribution = distribution / distribution.sum()  # Normalize to ensure it's a probability distribution
    distribution = distribution[distribution > 0]  # Remove zero probabilities
    entropy = -np.sum(distribution * np.log2(distribution))
    return entropy


def find_first_local_maximum(arr):
  """Finds local maxima in a 1D NumPy array, excluding edge points.

  Args:
    arr: A 1D NumPy array.

  Returns:
    The index of the first local maxima, if found, otherwise the last index.
  """
  for i in range(1, len(arr) - 1):
    if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
      return i
  return len(arr) - 1

# def calc_entropies_and_stds_per_delta(features, l_set, del_vals):
#     # calculate the entropy using running delta-s
#     entropy_val_lst = []
#     std_val_lst = []
#
#     for delta in tqdm(del_vals, desc="deltas"):
#         # construct the graph with the given delta
#         graph_df = build_graph(features, delta)
#
#         # remove edges that are connected to the labeled set
#         edge_from_seen = np.isin(graph_df.x, l_set)
#         covered_samples = graph_df.y[edge_from_seen].unique()
#         graph_df_pruned = graph_df[(~np.isin(graph_df.y, covered_samples))]
#
#         # calculate the degree of each vertex, and the entropy of the degree distribution
#         degrees = np.bincount(graph_df_pruned.x, minlength=len(features))
#         degrees_dist = np.bincount(degrees, minlength=degrees.max())
#         entropy = shannon_entropy(degrees_dist)
#         entropy_val_lst.append(entropy)
#         std_val_lst.append(np.std(degrees))
#
#         del graph_df, graph_df_pruned, degrees, degrees_dist
#
#     return entropy_val_lst, std_val_lst

def is_monotone_decreasing(arr):
    return all(arr[i] >= arr[i + 1] for i in range(len(arr) - 1))

def calc_entropies_and_stds_per_delta(features, l_set, del_vals, cosine_dist):
    # calculate the entropy using running delta-s
    argsort = np.arange(len(del_vals))
    if not is_monotone_decreasing(del_vals):
        argsort = np.argsort(-del_vals)
        del_vals = np.asarray(del_vals)[argsort]
        reverse_mapping = np.empty_like(argsort)
        reverse_mapping[argsort] = np.arange(len(del_vals))
        argsort = reverse_mapping

    entropy_val_lst = []
    std_val_lst = []
    max_degree_lst = []

    graph_df = build_graph(features, del_vals[0], cosine_dist)

    for delta in tqdm(del_vals, desc="deltas"):
        # remove the edges which are larger than the current delta
        graph_df = graph_df[graph_df.d < delta]

        # remove edges that are connected to the labeled set
        edge_from_seen = np.isin(graph_df.x, l_set)
        covered_samples = graph_df.y[edge_from_seen].unique()
        graph_df_pruned = graph_df[(~np.isin(graph_df.y, covered_samples))]

        # calculate the degree of each vertex, and the entropy of the degree distribution
        degrees = np.bincount(graph_df_pruned.x, minlength=len(features))
        degrees_dist = np.bincount(degrees, minlength=degrees.max())
        entropy = shannon_entropy(degrees_dist)
        entropy_val_lst.append(entropy)
        std_val_lst.append(np.std(degrees))
        max_degree_lst.append(degrees.max())

        del graph_df_pruned, degrees, degrees_dist

    del graph_df

    entropy_val_lst = [entropy_val_lst[i] for i in argsort]
    std_val_lst = [std_val_lst[i] for i in argsort]
    max_degree_lst = [max_degree_lst[i] for i in argsort]
    return entropy_val_lst, std_val_lst, max_degree_lst


def calculate_new_delta_by_deg_std(features, l_set, current_delta):
    # calculate the entropy using running delta-s
    # del_val = np.arange(0.05, current_delta + 0.05, 0.05)
    del_val = np.arange(0.02, current_delta + 0.02, 0.02)
    del_val = [round(delta, 2) for delta in del_val]
    entropy_val_lst, std_val_lst = calc_entropies_and_stds_per_delta(features, l_set, del_val)
    relevant_metric = entropy_val_lst  # std_val_lst

    #  maximum

    # entropy_argmax = del_val[np.argmax(entropy_val_lst)].item()
    # print(f"Maximum in entropy delta: {entropy_argmax}")
    argmax_index = np.argmax(relevant_metric)
    std_argmax = del_val[argmax_index].item()
    std_argmax_softened = std_argmax
    print(f"Maximum in std delta: {std_argmax}")

    #  local maximum

    # entropy_argmax = del_val[find_first_local_maximum(entropy_val_lst)].item()
    # print(f"Local maximum in entropy delta: {entropy_argmax}")

    # entropy_argmax = del_val[find_first_local_maxima(std_val_lst)].item()
    # print(f"Local maximum in std delta: {entropy_argmax}")

    # softening
    # score_new_delta = (std_max - std_val_lst[-1]) / std_max
    # std_argmax_softened = score_new_delta * std_argmax + (1 - score_new_delta) * current_delta
    # std_argmax_softened = (std_argmax + current_delta) / 2
    score_new_delta = (relevant_metric[argmax_index] - relevant_metric[-1]) / relevant_metric[argmax_index]
    std_argmax_softened = score_new_delta * std_argmax + (1 - score_new_delta) * current_delta
    std_argmax_softened = float(std_argmax_softened)
    print(f"Softened delta: {std_argmax_softened}")

    return std_argmax_softened
    # return current_delta - 0.05

def stable_logsumexp_softmax(values, temperature: float = 1.0):
    values = np.asarray(values).astype(float) / temperature
    log_probs = values - logsumexp(values)
    softmax_vector = np.exp(log_probs)
    if np.sum(softmax_vector) != 1.0:
        softmax_vector /= np.sum(softmax_vector)
    return softmax_vector

def cosine_similarity(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    x1_norm = torch.norm(x1, p=2, dim=1, keepdim=True)
    x2_norm = torch.norm(x2, p=2, dim=1, keepdim=True)
    x1_normalized = x1 / x1_norm
    x2_normalized = x2 / x2_norm
    cosine_similarity_matrix = (x1_normalized @ x2_normalized.T)
    return cosine_similarity_matrix


