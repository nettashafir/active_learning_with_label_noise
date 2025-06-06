"""
Taken from the official implementation of the DIRECT algorithm:

https://github.com/EfficientTraining/LabelBench/blob/main/LabelBench/strategy/strategy_impl/direct.py

Nuggehalli, S., Zhang, J., Jain, L., and Nowak, R. Direct: Deep active learning under imbalance and label noise (2023)
"""

import numpy as np
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
import pycls.utils.logging as lu
from tqdm import tqdm

logger = lu.get_logger(__name__)


class Node:
    def __init__(self, idx, pred, label):
        self.idx = idx
        self.pred = pred
        self.labeled = False
        self.label = label

    def update(self):
        assert not self.labeled
        self.labeled = True


class DIRECT:
    def __init__(self, cfg, lSet, uSet, budgetSize, dataObj):
        """
        Initialize DIRECT active learner

        Args:
            cfg: Config object with ACTIVE_LEARNING attributes
            lSet: Initial labeled set indices
            uSet: Initial unlabeled set indices
            budgetSize: Number of samples to select
        """
        self.cfg = cfg
        self.lSet = lSet
        self.uSet = uSet
        self.budgetSize = budgetSize
        self.dataObj = dataObj
        self.B1 = 5  # cfg.ACTIVE_LEARNING.B1 # Number of parallel annotators
        self.min_budget_per_class = 1  # cfg.ACTIVE_LEARNING.MIN_BUDGET_PER_CLASS

    @staticmethod
    def get_acc(w):
        cumsum = np.insert(np.cumsum(w), 0, 0)
        reverse_cumsum = np.insert(np.cumsum(w[::-1]), 0, 0)[::-1]
        acc = cumsum - reverse_cumsum
        return acc

    def find_best_hyp(self, w, n):
        acc = self.get_acc(w)
        best_hyp = np.argmax(acc)
        return best_hyp

    def version_space_reduction(self, graph, query_idxs, I, J, B1, B2, n, w, k, spend):
        """Version space reduction step of DIRECT algorithm"""
        print(f"DIRECT | Version space bounds - I: {I}, J: {J}")

        if B2 == 0:
            return spend, query_idxs

        budget = min(B1, B2)
        lmbda = np.zeros(n)
        lmbda[I:J + 1] = 1.0 / (J - I + 1)

        # Create lambda for unlabeled points
        lmbda_unlabeled = []
        for i, node in enumerate(graph):
            if node.labeled:
                lmbda_unlabeled.append(0)
            else:
                lmbda_unlabeled.append(lmbda[i])

        if sum(lmbda_unlabeled) == 0:
            return spend, query_idxs

        greater_than_zero = [num for num in lmbda_unlabeled if num > 0]
        num_sample = min(budget, len(greater_than_zero))

        if num_sample == 0:
            return spend, query_idxs

        samp_idxs = np.random.choice(n, size=num_sample, replace=False, p=np.array(lmbda_unlabeled) / sum(lmbda_unlabeled))

        for idx in samp_idxs:
            if not graph[idx].labeled:
                query_idxs.add(graph[idx].idx)
                spend += 1
                graph[idx].update()
                w[idx] = 1 if np.argmax(graph[idx].label) == k else -1

        # Compute diameter of uncertainty region
        cur_diam = sum(1 for m in range(I, J + 1) if not graph[m].labeled)

        if cur_diam > 0:
            base = B2 // B1
            c2 = cur_diam ** (1 / (B2 // B1)) if base > 1 else 1
            diam = cur_diam

            while diam > max(1, cur_diam / c2):
                accs = self.get_acc(w)
                if accs[I] < accs[J]:
                    if not graph[I].labeled:
                        diam -= 1
                    I += 1
                elif accs[I] > accs[J]:
                    if not graph[J].labeled:
                        diam -= 1
                    J -= 1
                else:
                    if np.random.randint(2) == 0:
                        if not graph[I].labeled:
                            diam -= 1
                        I += 1
                    else:
                        if not graph[J].labeled:
                            diam -= 1
                        J -= 1

        return self.version_space_reduction(graph, query_idxs, I, J, B1, B2 - budget, n, w, k, spend)

    def _compute_model_predictions(self, model, train_dataset, batch_size=100):
        """Compute model predictions on training dataset"""
        model.cuda()
        model.eval()
        model_predictions = []

        # Create a dataloader for all relevant indices
        # relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
        # dataloader = self.dataObj.getSequentialDataLoader(
        #     indexes=relevant_indices,
        #     batch_size=batch_size,
        #     data=train_dataset
        # )
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for inputs, _, _, _ in dataloader:
                inputs = inputs.cuda().type(torch.cuda.FloatTensor)
                outputs = model(inputs)
                softmax = torch.softmax(outputs, dim=1)
                model_predictions.append(softmax.cpu().numpy())

        model_predictions = np.concatenate(model_predictions, axis=0)
        print(f"DIRECT | Got model predictions with shape {model_predictions.shape}")
        return model_predictions

    def select_samples(self, model, train_dataset):
        """Main DIRECT sample selection method"""
        # Sanity check for initial sets
        assert len(set(self.lSet) & set(self.uSet)) == 0, "DIRECT | Overlap between labeled and unlabeled sets"

        # Get model predictions
        model_predictions = self._compute_model_predictions(model, train_dataset)
        labels_one_hot = np.zeros_like(model_predictions)

        # labels = np.argmax(model_predictions, axis=1)
        # labels_one_hot[np.arange(model_predictions.shape[0]), labels] = 1

        # labels = train_dataset.noisy_labels[self.lSet]
        # labels_one_hot[self.lSet, labels] = 1

        labels = train_dataset.noisy_labels
        labels_one_hot[np.arange(model_predictions.shape[0]), labels] = 1

        n_class = self.cfg.MODEL.NUM_CLASSES

        # Initialize nodes
        nodes = []
        margins = model_predictions - np.max(model_predictions, axis=1, keepdims=True)
        for idx, (margin, label) in enumerate(zip(margins, labels_one_hot)):
            nodes.append(Node(idx, margin, label)) # Labels will be filled when queried

        # Mark labeled samples
        class_freq = np.zeros(n_class)
        labeled_set = set(self.lSet)
        for i in labeled_set:
            nodes[i].update()
            class_freq[np.argmax(nodes[i].label)] += 1

        # Create sorted graphs for each class
        graphs = []
        reverse_graphs = []
        for i in range(n_class):
            sort_idx = np.argsort(-margins[:, i])
            graphs.append([nodes[idx] for idx in sort_idx])
            reverse_graphs.append([nodes[idx] for idx in sort_idx[::-1]])

        query_idxs = set()
        version_spend = 0
        class_rank = np.argsort(-class_freq)

        # Calculate effective classes and budget
        if self.budgetSize // n_class // 2 < self.min_budget_per_class:
            eff_class = n_class
            B2 = self.budgetSize // n_class // 2
        else:
            eff_class = self.budgetSize // (2 * self.min_budget_per_class)
            B2 = self.min_budget_per_class

        print(f"DIRECT | Effective classes: {eff_class}, Budget per class (B2): {B2}")

        # Version space reduction
        for i, k in enumerate(class_rank[-eff_class:]):
            graph = graphs[k]
            reverse_graph = reverse_graphs[k]

            w = []
            for node in graph:
                if not node.labeled:
                    w.append(0)
                elif np.argmax(node.label) == k:
                    w.append(1)
                else:
                    w.append(-1)

            # Find uncertainty region bounds
            I = J = 0
            for idx, node in enumerate(graph):
                # I* must be labeled
                if not node.labeled:
                    continue
                # I* is found
                if np.argmax(node.label) != k:
                    break
                I = idx
            for idx, node in enumerate(reverse_graph):
                # J* must be labeled
                if not node.labeled:
                    continue
                # J* is found
                if np.argmax(node.label) == k:
                    break
                J = idx
            J = len(graph) - J - 1
            assert I <= J, f"DIRECT | Invalid version space bounds: I ({I}) > J ({J})"

            # Adjust B2 based on unlabeled examples
            unlabeled = sum(1 for node in graph if not node.labeled)
            B2 = min(B2, unlabeled)

            spend, query_idxs = self.version_space_reduction(
                graph, query_idxs, I, J, self.B1, B2, len(graph),
                np.array(w), k, 0)
            version_spend += spend

        print(f"DIRECT | Version space reduction spent {version_spend} budget")

        # Query around hypothesis
        rem_budget = self.budgetSize - version_spend
        print(f"DIRECT | Starting hypothesis queries with remaining budget: {rem_budget}")

        for i, k in tqdm(enumerate(np.random.permutation(n_class)), desc="DIRECT | selecting samples"):
            graph = graphs[k]
            w = []
            for node in graph:
                if not node.labeled:
                    w.append(0)
                elif np.argmax(node.label) == k:
                    w.append(1)
                else:
                    w.append(-1)

            best_hyp = self.find_best_hyp(w, len(graph))
            right = best_hyp
            left = right - 1
            bit = 1
            left_exceed = right_exceed = False

            class_budget = rem_budget * (i + 1) // n_class - rem_budget * i // n_class
            while class_budget > 0:
                bit = (bit + 1) % 2
                if left_exceed and right_exceed:
                    break
                if bit == 0:
                    if left < 0:
                        left_exceed = True
                        continue
                    if not graph[left].labeled:
                        query_idxs.add(graph[left].idx)
                        graph[left].update()
                        class_budget -= 1
                    left -= 1
                else:
                    if right >= len(nodes):
                        right_exceed = True
                        continue
                    if not graph[right].labeled:
                        query_idxs.add(graph[right].idx)
                        graph[right].update()
                        class_budget -= 1
                    right += 1

        activeSet = list(query_idxs)
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))

        # Final sanity checks
        print(f"DIRECT | Selected {len(activeSet)} samples in active set")
        assert len(
            activeSet) <= self.budgetSize, f"DIRECT | Selected more samples ({len(activeSet)}) than budget ({self.budgetSize})"
        assert len(set(activeSet) & set(self.lSet)) == 0, "DIRECT | Active set contains samples from labeled set"
        assert len(set(activeSet) & set(remainSet)) == 0, "DIRECT | Overlap between active set and remain set"

        return activeSet, remainSet