import numpy as np
import pandas as pd
import torch
import pycls.datasets.utils as ds_utils
import pycls.utils.logging as lu
from tqdm import tqdm

try:
    from al_utils import construct_graph_for_large_datasets, cosine_similarity
except ImportError:
    from .al_utils import construct_graph_for_large_datasets, cosine_similarity

logger = lu.get_logger(__name__)


class ProbCover:
    def __init__(self, cfg, lSet, uSet, budgetSize, delta_scheduler):
        self.cfg = cfg
        self.seed = self.cfg['RNG_SEED']
        self.ds_name = self.cfg['DATASET']['NAME']
        self.representation_model = self.cfg['DATASET']['REPRESENTATION_MODEL']
        self.all_features = ds_utils.load_features(self.ds_name, representation_model=self.representation_model, train=True, normalize=True, project=cfg.ACTIVE_LEARNING.PROJECT_FEATURES_TO_UNIT_SPHERE)

        self.delta_scheduler = delta_scheduler
        self.lSet = lSet
        self.uSet = uSet
        self.budgetSize = budgetSize
        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
        self.rel_features = self.all_features[self.relevant_indices]
        self.graph_df = self.construct_graph()

    def construct_graph(self, batch_size=500):
        """
        creates a directed graph where:
        x->y iff l2(x,y) < delta.

        represented by a list of edges (a sparse matrix).
        stored in a dataframe
        """
        if len(self.rel_features) > 100_000:
            return construct_graph_for_large_datasets(self.rel_features, self.delta_scheduler.current_delta, batch_size)
        xs, ys, ds = [], [], []
        distance_measure = 'cosine' if self.cfg.ACTIVE_LEARNING.USE_COSINE_DIST else 'euclidean'
        print(f'ProbCover | Start constructing graph using delta={self.delta_scheduler.current_delta} using {distance_measure} distance.')
        logger.info(f'ProbCover | Start constructing graph using delta={self.delta_scheduler.current_delta} using {distance_measure} distance.')
        # distance computations are done in GPU
        cuda_feats = torch.tensor(self.rel_features).cuda()
        for i in tqdm(range(len(self.rel_features) // batch_size)):
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
        print(f'ProbCover | Finished constructing graph using delta={self.delta_scheduler.current_delta} using {distance_measure} distance.')
        print(f'ProbCover | Graph contains {len(df)} edges.')
        logger.info(f'ProbCover | Finished constructing graph using delta={self.delta_scheduler.current_delta} using {distance_measure} distance.')
        logger.info(f'ProbCover | Graph contains {len(df)} edges.')
        return df

    def select_samples(self):
        """
        selecting samples using the greedy algorithm.
        iteratively:
        - removes incoming edges to all covered samples
        - selects the sample high the highest out degree (covers most new samples)

        """
        print(f'ProbCover | Start selecting {self.budgetSize} samples.')
        logger.info(f'ProbCover | Start selecting {self.budgetSize} samples.')
        selected = []
        # removing incoming edges to all covered samples from the existing labeled set
        edge_from_seen = np.isin(self.graph_df.x, np.arange(len(self.lSet)))
        covered_samples = self.graph_df.y[edge_from_seen].unique()
        cur_df = self.graph_df[(~np.isin(self.graph_df.y, covered_samples))]
        max_degree = np.inf
        for i in range(self.budgetSize):
            curr_l_set = np.concatenate((np.arange(len(self.lSet)), selected)).astype(int)

            # Update the delta
            num_labeled_samples = len(self.lSet) + len(selected)
            if num_labeled_samples > 0 and num_labeled_samples % self.cfg.MODEL.NUM_CLASSES == 0:
                curr_delta = self.delta_scheduler.current_delta
                new_delta = self.delta_scheduler.update_delta(self.relevant_indices[curr_l_set], current_max_deg=max_degree)
                if new_delta != curr_delta:
                    self.graph_df = self.construct_graph()
                    edge_from_seen = np.isin(self.graph_df.x, np.asarray(curr_l_set))
                    covered_samples = self.graph_df.y[edge_from_seen].unique()
                    cur_df = self.graph_df[(~np.isin(self.graph_df.y, covered_samples))]


            coverage = len(covered_samples) / len(self.relevant_indices)
            # selecting the sample with the highest degree
            degrees = np.bincount(cur_df.x, minlength=len(self.relevant_indices))
            max_degree = np.max(degrees)
            degrees[curr_l_set] = -1  # remove already labeled samples

            print(f'ProbCover | Iteration is {i}.\tLabeled set size - {len(curr_l_set)}.\tGraph has {len(cur_df)} edges.\tMax degree is {max_degree}.\tCoverage is {coverage:.3f}')
            logger.info(f'ProbCover | Iteration is {i}.\tLabeled set size - {len(curr_l_set)}.\tGraph has {len(cur_df)} edges.\tMax degree is {max_degree}.\tCoverage is {coverage:.3f}')

            # Take the max degree
            # cur = degrees.argmax()
            # cur = np.random.choice(degrees.argsort()[::-1][:5]) # the paper randomizes selection
            max_degree = np.max(degrees)
            agrmax = np.arange(len(degrees))[degrees == max_degree]
            cur = np.random.choice(agrmax)

            # removing incoming edges to newly covered samples
            new_covered_samples = cur_df.y[(cur_df.x == cur)].values
            assert len(np.intersect1d(covered_samples, new_covered_samples)) == 0, 'ProbCover | all samples should be new'
            cur_df = cur_df[(~np.isin(cur_df.y, new_covered_samples))]

            covered_samples = np.concatenate([covered_samples, new_covered_samples])
            selected.append(cur)

        assert len(selected) == self.budgetSize, 'ProbCover | added a different number of samples'
        activeSet = self.relevant_indices[selected]
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))

        print(f'ProbCover | Finished the selection of {len(activeSet)} samples.')
        print(f'ProbCover | Active set is {activeSet}')
        logger.info(f'ProbCover | Finished the selection of {len(activeSet)} samples.')
        logger.info(f'ProbCover | Active set is {activeSet}')
        return activeSet, remainSet
