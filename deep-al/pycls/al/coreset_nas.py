import numpy as np
import torch
from torch.utils.data import DataLoader
import time
import math
from tqdm import tqdm
import pycls.utils.logging as lu

logger = lu.get_logger(__name__)

class CoreSetNas:
    """
    Implements coreset MIP sampling operation
    """

    def __init__(self, cfg, dataObj, lnl_obj, isMIP=False):
        self.dataObj = dataObj
        self.cuda_id = torch.cuda.current_device()
        self.cfg = cfg
        self.lnl_obj = lnl_obj
        self.isMIP = isMIP

    def _divide_budget(self):
        """Divides total budget into shares of size num_classes"""
        b = self.cfg.ACTIVE_LEARNING.BUDGET_SIZE
        c = self.cfg.MODEL.NUM_CLASSES

        if self.cfg.NOISE.FILTERING_FN.lower() == "ideal":
            budget_shares = [1] * b
        else:
            n = b // c
            remainder = b % c
            budget_shares = [c] * n
            if remainder > 0:
                budget_shares[0] += remainder

        return budget_shares

    @torch.no_grad()
    def get_representation(self, clf_model, dataset):
        clf_model.cuda(self.cuda_id)
        dataloader = DataLoader(dataset, batch_size=int(self.cfg.TRAIN.BATCH_SIZE/self.cfg.NUM_GPUS), shuffle=False, num_workers=4)
        features = []

        print(f"len(dataLoader): {len(dataloader)}")

        for i, (x, _, _, _) in enumerate(tqdm(dataloader, desc="Extracting Representations")):
            with torch.no_grad():
                x = x.cuda(self.cuda_id)
                x = x.type(torch.cuda.FloatTensor)
                temp_z, _ = clf_model(x)
                features.append(temp_z.cpu().numpy())

        features = np.concatenate(features, axis=0)
        return features

    def gpu_compute_dists(self, M1, M2):
        """
        Computes L2 norm square on gpu
        Assume
        M1: M x D matrix
        M2: N x D matrix

        output: M x N matrix
        """
        # print(f"Function call to gpu_compute dists; M1: {M1.shape} and M2: {M2.shape}")
        M1_norm = (M1 ** 2).sum(1).reshape(-1, 1)

        M2_t = torch.transpose(M2, 0, 1)
        M2_norm = (M2 ** 2).sum(1).reshape(1, -1)
        dists = M1_norm + M2_norm - 2.0 * torch.mm(M1, M2_t)
        return dists

    def compute_dists(self, X, X_train):
        dists = -2 * np.dot(X, X_train.T) + np.sum(X_train ** 2, axis=1) + np.sum(X ** 2, axis=1).reshape((-1, 1))
        return dists

    def optimal_greedy_k_center(self, labeled, unlabeled):
        n_lSet = labeled.shape[0]
        lSetIds = np.arange(n_lSet)
        n_uSet = unlabeled.shape[0]
        uSetIds = n_lSet + np.arange(n_uSet)

        # order is important
        features = np.vstack((labeled, unlabeled))
        print("Started computing distance matrix of {}x{}".format(features.shape[0], features.shape[0]))
        start = time.time()
        distance_mat = self.compute_dists(features, features)
        end = time.time()
        print("Distance matrix computed in {} seconds".format(end - start))
        greedy_indices = []
        for i in range(self.cfg.ACTIVE_LEARNING.BUDGET_SIZE):
            if i != 0 and i % 500 == 0:
                print("Sampled {} samples".format(i))
            lab_temp_indexes = np.array(np.append(lSetIds, greedy_indices), dtype=int)
            min_dist = np.min(distance_mat[lab_temp_indexes, n_lSet:], axis=0)
            active_index = np.argmax(min_dist)
            greedy_indices.append(n_lSet + active_index)

        remainSet = set(np.arange(features.shape[0])) - set(greedy_indices) - set(lSetIds)
        remainSet = np.array(list(remainSet))

        return greedy_indices - n_lSet, remainSet

    def greedy_k_center(self, labeled, unlabeled, budget_size):
        greedy_indices = [None for i in range(budget_size)]
        greedy_indices_counter = 0
        #move cpu to gpu
        labeled = torch.from_numpy(labeled).cuda(0)
        unlabeled = torch.from_numpy(unlabeled).cuda(0)

        print(f"[GPU] Labeled.shape: {labeled.shape}")
        print(f"[GPU] Unlabeled.shape: {unlabeled.shape}")
        # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
        st = time.time()
        min_dist,_ = torch.min(self.gpu_compute_dists(labeled[0,:].reshape((1,labeled.shape[1])), unlabeled), dim=0)
        min_dist = torch.reshape(min_dist, (1, min_dist.shape[0]))
        print(f"time taken: {time.time() - st} seconds")

        temp_range = 500
        dist = np.empty((temp_range, unlabeled.shape[0]))
        for j in tqdm(range(1, labeled.shape[0], temp_range), desc="Getting first farthest index"):
            if j + temp_range < labeled.shape[0]:
                dist = self.gpu_compute_dists(labeled[j:j+temp_range, :], unlabeled)
            else:
                dist = self.gpu_compute_dists(labeled[j:, :], unlabeled)

            min_dist = torch.cat((min_dist, torch.min(dist,dim=0)[0].reshape((1,min_dist.shape[1]))))

            min_dist = torch.min(min_dist, dim=0)[0]
            min_dist = torch.reshape(min_dist, (1, min_dist.shape[0]))

        # iteratively insert the farthest index and recalculate the minimum distances:
        _, farthest = torch.max(min_dist, dim=1)
        greedy_indices[greedy_indices_counter] = farthest.item()
        greedy_indices_counter += 1

        amount = budget_size - 1

        for i in tqdm(range(amount), desc = "Constructing Active set"):
            dist = self.gpu_compute_dists(unlabeled[greedy_indices[greedy_indices_counter-1], :].reshape((1,unlabeled.shape[1])), unlabeled)

            min_dist = torch.cat((min_dist, dist.reshape((1, min_dist.shape[1]))))

            min_dist, _ = torch.min(min_dist, dim=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            _, farthest = torch.max(min_dist, dim=1)
            greedy_indices[greedy_indices_counter] = farthest.item()
            greedy_indices_counter += 1

        remainSet = set(np.arange(unlabeled.shape[0])) - set(greedy_indices)
        remainSet = np.array(list(remainSet))
        if self.isMIP:
            return greedy_indices,remainSet,math.sqrt(np.max(min_dist))
        else:
            return greedy_indices, remainSet

    def query(self, lSet, uSet, clf_model, dataset):
        """Main query function that handles noisy label detection and batched sampling"""
        assert not clf_model.training, "Classification model expected in evaluation mode"
        assert clf_model.penultimate_active, "Classification model expected in penultimate mode"

        # Initialize sets
        initial_l_set = np.asarray(lSet)
        initial_u_set = np.asarray(uSet)

        # Get all representations at once
        print("Extracting representations for all points")
        all_representations = self.get_representation(clf_model, dataset)

        curr_l_set = initial_l_set.astype(int)
        curr_u_set = initial_u_set.astype(int)
        budget_shares = self._divide_budget()
        selected = []

        # Process each budget share
        for share_idx, share in enumerate(budget_shares):
            print(f"Processing budget share {share_idx + 1}/{len(budget_shares)} of size {share}")

            # Identify noisy samples in current labeled set
            if len(initial_l_set) == 0:
                curr_l_set_clean = []
                curr_l_set_noisy = []
                curr_l_set_is_clean = np.array([])
            else:
                curr_l_set_is_clean, _, _ = self.lnl_obj.identify_noisy_samples(l_set=curr_l_set, u_set=curr_u_set)
                curr_l_set_clean = curr_l_set[curr_l_set_is_clean]
                curr_l_set_noisy = curr_l_set[~curr_l_set_is_clean]

            clean_lb_repr = all_representations[curr_l_set_clean]
            lb_repr = all_representations[curr_l_set]
            ul_repr = all_representations[curr_u_set]

            # Select samples using k-center greedy
            print(f"Selecting {share} samples using Coreset")
            greedy_indices, remain_indices = self.greedy_k_center(clean_lb_repr, ul_repr, share)

            selected_batch = curr_u_set[greedy_indices]
            assert len(np.intersect1d(curr_l_set_clean, selected_batch)) == 0, 'all samples should be new'
            assert len(np.intersect1d(curr_l_set_noisy, selected_batch)) == 0, 'all samples should be new'
            selected.extend(selected_batch)

            # Update sets for next iteration
            curr_l_set = np.concatenate([curr_l_set, selected_batch]).astype(int)
            curr_u_set = curr_u_set[remain_indices].astype(int)

        active_set = np.array(selected)
        remain_set = curr_u_set

        assert len(selected) == self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, 'added a different number of samples'
        assert len(np.intersect1d(lSet, active_set)) == 0, 'all samples should be new'

        return active_set, remain_set