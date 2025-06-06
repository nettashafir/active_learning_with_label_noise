import numpy as np
import pandas as pd
import torch
from copy import deepcopy

import pycls.datasets.utils as ds_utils
import pycls.utils.logging as lu
from al_utils import cosine_similarity, stable_logsumexp_softmax

logger = lu.get_logger(__name__)


class NPC:
    def __init__(self, cfg, l_set_clean, l_set_noisy, u_set, budget_size, delta_scheduler, lnl_obj=None):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        self.representation_model = self.cfg['DATASET']['REPRESENTATION_MODEL']
        self.all_features = ds_utils.load_features(self.ds_name, representation_model=self.representation_model, train=True, normalize=True, project=cfg.ACTIVE_LEARNING.PROJECT_FEATURES_TO_UNIT_SPHERE)
        self.delta_scheduler = delta_scheduler
        self.u_set = u_set
        self.l_set_clean = l_set_clean
        self.l_set_noisy = l_set_noisy
        self.budget_size = budget_size
        self.relevant_indices = np.concatenate([self.l_set_clean, self.l_set_noisy, self.u_set]).astype(int)
        self.rel_features = self.all_features[self.relevant_indices]
        self.lnl_obj = lnl_obj
        self.full_graph = self.construct_graph()

    def _divide_budget(self):
        """Divides total budget into shares of size num_classes"""
        b = self.budget_size
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

    def construct_graph(self, batch_size=500):
        """
        creates a directed graph where:
        x->y iff l2(x,y) < delta.

        represented by a list of edges (a sparse matrix).
        stored in a dataframe
        """
        xs, ys, ds = [], [], []
        distance_measure = 'cosine' if self.cfg.ACTIVE_LEARNING.USE_COSINE_DIST else 'euclidean'
        print(f'NPC | Start constructing graph using delta={self.delta_scheduler.current_delta} using {distance_measure} distance.')
        logger.info(f'NPC | Start constructing graph using delta={self.delta_scheduler.current_delta} using {distance_measure} distance.')
        # distance computations are done in GPU
        cuda_feats = torch.tensor(self.rel_features).cuda()
        for i in range(len(self.rel_features) // batch_size):
            # distance comparisons are done in batches to reduce memory consumption
            cur_feats = cuda_feats[i * batch_size: (i + 1) * batch_size]
            if self.cfg.ACTIVE_LEARNING.USE_COSINE_DIST:
                dist = (1 - cosine_similarity(cur_feats, cuda_feats))
            else:
                dist = torch.cdist(cur_feats, cuda_feats)
            mask = dist < self.delta_scheduler.current_delta
            # saving edges using indices list - saves memory.
            x, y = mask.nonzero().T
            xs.append(x.cpu() + batch_size * i)
            ys.append(y.cpu())
            ds.append(dist[mask].cpu())

        xs = torch.cat(xs).numpy()
        ys = torch.cat(ys).numpy()
        ds = torch.cat(ds).numpy()

        df = pd.DataFrame({'x': xs, 'y': ys, 'd': ds})
        print(f'NPC | Finished constructing graph using delta={self.delta_scheduler.current_delta} using {distance_measure} distance.')
        print(f'NPC | Graph contains {len(df)} edges.')
        logger.info(f'NPC | Finished constructing graph using delta={self.delta_scheduler.current_delta} using {distance_measure} distance.')
        logger.info(f'NPC | Graph contains {len(df)} edges.')
        return df

    def select_samples(self):
        """
        selecting samples using the greedy algorithm.
        iteratively:
        - removes incoming edges to all covered samples
        - selects the sample high the highest out degree (covers most new samples)
        """

        initial_l_set_clean = np.arange(len(self.l_set_clean))
        initial_l_set_noisy = np.arange(len(self.l_set_clean), len(self.l_set_clean) + len(self.l_set_noisy))
        initial_l_set = np.concatenate([initial_l_set_clean, initial_l_set_noisy])
        initial_u_set = np.arange(len(self.l_set_clean) + len(self.l_set_noisy), len(self.relevant_indices))

        curr_l_set_clean = initial_l_set_clean
        curr_l_set_noisy = initial_l_set_noisy
        curr_l_set = initial_l_set
        curr_l_set_is_clean = np.concatenate([np.full(len(curr_l_set_clean), True),
                                               np.full(len(curr_l_set_noisy), False)]).astype(bool)
        curr_u_set = initial_u_set

        print(f'NPC | Start selecting {self.budget_size} samples.')
        logger.info(f'NPC | Start selecting {self.budget_size} samples.')
        budget_shares = self._divide_budget()


        cur_df = None
        it = 0
        selected = []
        selected_j = []
        max_degree = np.inf
        for j, share in enumerate(budget_shares):
            if self.cfg.NOISE.FILTERING_FN.lower() != "ideal" or j == 0:
                cur_df = deepcopy(self.full_graph)
                remove_from_graph = curr_l_set_clean
            else:
                remove_from_graph = np.intersect1d(curr_l_set_clean, selected_j)  # Only a single sample is added

            # removing incoming edges to all covered samples from the existing clean labeled set
            edge_from_seen = np.isin(cur_df.x, remove_from_graph)
            covered_samples = cur_df.y[edge_from_seen].unique()
            cur_df = cur_df[(~np.isin(cur_df.y, covered_samples))]

            # removing outgoing edges from the noisy labeled set
            cur_df = cur_df[(~np.isin(cur_df.x, curr_l_set_noisy))]

            selected_j = []
            for i in range(share):
                curr_l_set = np.concatenate([initial_l_set, selected, selected_j]).astype(int)

                # Update the delta
                num_labeled_samples = len(curr_l_set)
                if num_labeled_samples > 0 and num_labeled_samples % self.cfg.MODEL.NUM_CLASSES == 0:
                    curr_delta = self.delta_scheduler.current_delta
                    curr_l_set_is_clean = np.concatenate([curr_l_set_is_clean, np.full(len(selected_j), True)]).astype(bool)
                    new_delta = self.delta_scheduler.update_delta(self.relevant_indices[curr_l_set], curr_l_set_is_clean, current_max_deg=max_degree)
                    if new_delta != curr_delta:
                        self.full_graph = cur_df = self.construct_graph()
                        edge_from_seen = np.isin(cur_df.x, curr_l_set_clean)
                        covered_samples = cur_df.y[edge_from_seen].unique()
                        cur_df = cur_df[(~np.isin(cur_df.y, covered_samples))]
                        cur_df = cur_df[(~np.isin(cur_df.x, curr_l_set_noisy))]

                coverage = (len(covered_samples) + len(curr_l_set_noisy)) / len(self.relevant_indices)
                # selecting the sample with the highest degree
                degrees = np.bincount(cur_df.x, minlength=len(self.relevant_indices))
                # print(f"NPC | Degrees distribution:\n{np.bincount(degrees)}")

                # Take the max degree
                degrees[curr_l_set] = -1
                max_degree = np.max(degrees)
                agrmax = np.arange(len(degrees))[degrees == max_degree]
                cur = np.random.choice(agrmax)

                print(f'NPC | Iteration is {it}.\tLabeled set size - {len(curr_l_set)}.\tGraph has {len(cur_df)} edges.\tMax degree is {degrees.max()}.\tCoverage is {coverage:.3f}')
                logger.info(f'NPC | Iteration is {it}.\tLabeled set size - {len(curr_l_set)}.\tGraph has {len(cur_df)} edges.\tMax degree is {degrees.max()}.\tCoverage is {coverage:.3f}')
                it += 1

                # removing incoming edges to newly covered samples
                new_covered_samples = cur_df.y[(cur_df.x == cur)].values
                # assert len(np.intersect1d(covered_samples, new_covered_samples)) == 0, 'all samples should be new'
                if self.cfg.NOISE.FILTERING_FN.lower() != "ideal":
                    cur_df = cur_df[(~np.isin(cur_df.y, new_covered_samples))]

                covered_samples = np.concatenate([covered_samples, new_covered_samples])
                selected_j.append(cur)

            selected.extend(selected_j)

            # update the clean and noisy sets
            lnl_method = self.cfg.NOISE.FILTERING_FN.lower()
            curr_l_set = np.concatenate([initial_l_set_clean, initial_l_set_noisy, selected]).astype(int)
            curr_u_set = np.array(list(set(curr_u_set) - set(curr_l_set))).astype(int)

            if lnl_method != "ideal" and len(curr_l_set) < self.cfg.MODEL.NUM_CLASSES:
                print("NPC | Not enough samples to train the LNL model. Keep the selection.")
                logger.info("NPC | Not enough samples to train the LNL model. Keep the selection.")
                curr_l_set_clean = np.concatenate((initial_l_set_clean, selected))
                continue

            curr_l_set_is_clean, _, _ = self.lnl_obj.identify_noisy_samples(l_set=self.relevant_indices[curr_l_set], u_set=self.relevant_indices[curr_u_set])
            curr_l_set_is_clean_true = self.lnl_obj.is_clean[self.relevant_indices[curr_l_set]]
            accuracy, clean_precision, clean_recall, clean_f1, noise_precision, noise_recall, noise_f1, true_noise, predicted_noise = \
                self.lnl_obj.calc_noise_metrics(curr_l_set_is_clean, curr_l_set_is_clean_true)
            if lnl_method != "ideal":
                noise_metrics_msg = f"ProbCover | budget : {len(curr_l_set)}, noise metrics:" \
                      f"\n\tTrue noise - {true_noise:.3f}" \
                      f"\n\tPredicted noise - {predicted_noise:.3f}" \
                      f"\n\taccuracy  - {accuracy:.3f}" \
                      f"\n\tclean_recall - {clean_recall:.3f}" \
                      f"\n\tclean_precision - {clean_precision:.3f}" \
                      f"\n\tclean_f1 - {clean_f1:.3f}"
                print(noise_metrics_msg)
                logger.info(noise_metrics_msg)

            # dropout
            if self.cfg.ACTIVE_LEARNING.NOISE_DROPOUT:
                predicted_clean_ratio = float(np.mean(curr_l_set_is_clean.astype(int)))
                eta = max(min(predicted_clean_ratio, 1 - predicted_clean_ratio), 0.1)
                print(f"NPC | Noise Dropout - Flip {eta} of the noisy samples to clean")
                logger.info(f"NPC | Noise Dropout - Flip {eta} of the noisy samples to clean")
                false_indices = np.where(curr_l_set_is_clean == False)[0]
                n_to_flip = int(len(false_indices) * eta)
                indices_to_flip = np.random.choice(false_indices, size=n_to_flip, replace=False)
                curr_l_set_is_clean[indices_to_flip] = True

            curr_l_set_clean = curr_l_set[curr_l_set_is_clean].astype(int)
            curr_l_set_noisy = curr_l_set[~curr_l_set_is_clean].astype(int)

        assert len(selected) == self.budget_size, 'NPC | added a different number of samples'
        assert len(np.intersect1d(initial_l_set, selected)) == 0, 'NPC | all samples should be new'
        active_set = self.relevant_indices[selected]
        remain_set = np.array(sorted(list(set(self.u_set) - set(active_set))))

        print(f'NPC | Finished the selection of {len(active_set)} samples.')
        print(f'NPC | Active set is {active_set}')
        logger.info(f'NPC | Finished the selection of {len(active_set)} samples.')
        logger.info(f'NPC | Active set is {active_set}')
        assert len(set(active_set) & set(self.l_set_clean)) == 0, 'NPC | active set should not contain labeled samples'
        assert len(set(active_set) & set(self.l_set_noisy)) == 0, 'NPC | active set should not contain noisy samples'
        return active_set, remain_set
