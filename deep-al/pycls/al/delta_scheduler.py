import numpy as np
from warnings import warn

from al_utils import calc_entropies_and_stds_per_delta, choose_delta_for_probcover
import pycls.datasets.utils as ds_utils
import pycls.utils.logging as lu


EXPONENTIAL_FUNCTION = lambda x, k: (np.exp(-k) - np.exp(-k * x)) / (np.exp(-k) - 1)

logger = lu.get_logger(__name__)

class DeltaScheduler:
    def __init__(self, cfg, features):
        """
        Initializes the DeltaScheduler.

        Parameters:
            cfg: Config object with ACTIVE_LEARNING attributes.
            features: MxD numpy array representing the dataset points in R^d.
        """
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.representation_model = self.cfg['DATASET']['REPRESENTATION_MODEL']
        self.features = ds_utils.load_features(self.ds_name, representation_model=self.representation_model, train=True, normalize=True, project=self.cfg.ACTIVE_LEARNING.USE_COSINE_DIST)
        max_delta = cfg.ACTIVE_LEARNING.MAX_DELTA
        delta_resolution = cfg.ACTIVE_LEARNING.DELTA_RESOLUTION
        self.delta_support = np.arange(delta_resolution, max_delta + delta_resolution, delta_resolution)
        self.initial_delta = self.cfg.ACTIVE_LEARNING.INITIAL_DELTA
        assert self.initial_delta is not None, "DeltaScheduler | Initial delta must be set in the config."
        self.delta_history = {self.initial_delta: (len(features), None)}
        self.current_delta: float = self.initial_delta


        # print the delta policy
        policy = self.cfg.ACTIVE_LEARNING.DELTA_POLICY.MAJOR_POLICY
        softening_policy = self.cfg.ACTIVE_LEARNING.DELTA_POLICY.SOFTENING_POLICY
        consider_noise = self.cfg.ACTIVE_LEARNING.DELTA_POLICY.CONSIDER_NOISE
        if softening_policy is not None and policy in ["constant", "linear_descent", "exponential_descent"]:
            warn(f"Delta softening policy is not applicable for {policy} delta policy. Ignoring delta softening policy.")
            self.cfg.ACTIVE_LEARNING.DELTA_POLICY.SOFTENING_POLICY = softening_policy =  None
        print(f"DeltaScheduler | Policy: {policy}. Softening policy: {softening_policy}. Consider noise: {consider_noise}")
        logger.info(f"DeltaScheduler | Policy: {policy}. Softening policy: {softening_policy}. Consider noise: {consider_noise}")


    def update_delta(self, l_set, l_set_predicted_is_clean=None, current_max_deg=None, targets=None):
        policy = self.cfg.ACTIVE_LEARNING.DELTA_POLICY.MAJOR_POLICY
        softening_policy = self.cfg.ACTIVE_LEARNING.DELTA_POLICY.SOFTENING_POLICY
        consider_noise = self.cfg.ACTIVE_LEARNING.DELTA_POLICY.CONSIDER_NOISE
        val_curr_delta = val_new_delta = None
        print(f"DeltaScheduler | Updating delta using policy: {policy}. Softening policy: {softening_policy}. Consider noise: {consider_noise}")
        logger.info(f"DeltaScheduler | Updating delta using policy: {policy}. Softening policy: {softening_policy}. Consider noise: {consider_noise}")

        # noise consideration
        assert not (consider_noise and l_set_predicted_is_clean is None), "If considering noise, l_set_predicted_is_clean must be provided."
        if consider_noise:
            l_set = l_set[l_set_predicted_is_clean]
            print(f"DeltaScheduler | Ignoring noisy samples. Number of samples after cleaning: {len(l_set)}")
            logger.info(f"DeltaScheduler | Ignoring noisy samples. Number of samples after cleaning: {len(l_set)}")

        # Find new delta based on the policy
        new_delta, delta_values, curr_support = self.choose_delta_from_policy(l_set, policy, current_max_deg)
        if delta_values is not None:
            curr_delta_index = np.where(curr_support == self.current_delta)[0][0]
            val_curr_delta = delta_values[curr_delta_index]
            new_delta_index = np.where(curr_support == new_delta)[0][0]
            val_new_delta = delta_values[new_delta_index]

        # Softening the delta based on the softening policy
        assert not (softening_policy is not None and policy in ["constant", "linear_descent", "exponential_descent"]), f"Softening policy {softening_policy} is not supported for this policy."

        if softening_policy is None:
            score_new_delta = 1
        elif softening_policy == "average":
            score_new_delta = 0.5
        elif softening_policy == "improvement_proportion":
            score_new_delta = (val_new_delta - val_curr_delta) / val_new_delta
        else:
            raise ValueError(f"DeltaScheduler | Unsupported softening policy: {softening_policy}")

        new_delta = score_new_delta * new_delta + (1-score_new_delta) * self.current_delta

        if new_delta != self.current_delta:
            print(f"DeltaScheduler | Delta updated from {round(self.current_delta, 5)} to {round(new_delta, 5)}")
            logger.info(f"DeltaScheduler | Delta updated from {round(self.current_delta, 5)} to {round(new_delta, 5)}")
        self.current_delta = new_delta
        self.delta_history[new_delta] = (len(l_set), delta_values)
        return new_delta


    def choose_delta_from_policy(self, l_set, policy, current_max_deg=None):
        """
        Compute the new delta based on the active learning policy.

        Returns:
            float, list: The new delta, and an array of the delta values.
        """
        values = None

        curr_support = np.asarray(self.delta_support)
        if self.current_delta not in curr_support:
            curr_support = np.concatenate([curr_support, [self.current_delta]])
            curr_support = np.sort(curr_support)
        curr_support = curr_support[::-1]

        use_cosine_dist = self.cfg.ACTIVE_LEARNING.USE_COSINE_DIST

        if policy == "constant":
            delta = self.current_delta
        elif policy == "linear_descent":
            proportion = len(l_set) / len(self.features)
            coeff = 1 - proportion
            delta = coeff * self.initial_delta + (1-coeff) * (self.delta_support[0])
        elif policy == "exponential_descent":
            proportion = len(l_set) / len(self.features)
            k = self.cfg.ACTIVE_LEARNING.DELTA_POLICY.EXPONENTIAL_K
            coeff = EXPONENTIAL_FUNCTION(proportion, k)
            delta = coeff * self.initial_delta + (1-coeff) * (self.delta_support[0])
        elif policy == "max_std":
            _, values, _ = calc_entropies_and_stds_per_delta(self.features, l_set, curr_support, use_cosine_dist)
            argmax = np.argmax(values)
            delta = curr_support[argmax].item()
            print(f"DeltaScheduler | Maximum std delta: {round(delta, 5)}")
            logger.info(f"DeltaScheduler | Maximum std delta: {round(delta, 5)}")
        elif policy == "max_std_smaller":
            assert self.current_delta is not None, "Current delta must be set before using max_std_smaller policy."
            curr_support = curr_support[curr_support <= self.current_delta]
            _, values, _ = calc_entropies_and_stds_per_delta(self.features, l_set, curr_support, use_cosine_dist)
            argmax = np.argmax(values)
            delta = curr_support[argmax].item()
            print(f"DeltaScheduler | Maximum std delta smaller than current delta: {round(delta, 5)}")
            logger.info(f"DeltaScheduler | Maximum std delta smaller than current delta: {round(delta, 5)}")
        elif policy == "max_entropy":
            values, _, _ = calc_entropies_and_stds_per_delta(self.features, l_set, curr_support, use_cosine_dist)
            argmax = np.argmax(values)
            delta = curr_support[argmax].item()
            print(f"DeltaScheduler | Maximum entropy delta: {round(delta, 5)}")
            logger.info(f"DeltaScheduler | Maximum entropy delta: {round(delta, 5)}")
        elif policy == "max_entropy_smaller":
            assert self.current_delta is not None, "Current delta must be set before using max_entropy_smaller policy."
            curr_support = curr_support[curr_support <= self.current_delta]
            values, _, _ = calc_entropies_and_stds_per_delta(self.features, l_set, curr_support, use_cosine_dist)
            argmax = np.argmax(values)
            delta = curr_support[argmax].item()
            print(f"DeltaScheduler | Maximum entropy delta smaller than current delta: {round(delta, 5)}")
            logger.info(f"DeltaScheduler | Maximum entropy delta smaller than current delta: {round(delta, 5)}")
        elif policy == "max_support_size_take_max":
            _, _, values = calc_entropies_and_stds_per_delta(self.features, l_set, curr_support, use_cosine_dist)
            argmax = np.argmax(values)
            delta = curr_support[argmax].item()
            print(f"DeltaScheduler | Max support size: {np.max(values)}, argmax delta: {round(delta, 5)}")
            logger.info(f"DeltaScheduler | Max support size: {np.max(values)}, argmax delta: {round(delta)}")
        elif policy == "max_support_size_take_min":
            _, _, values = calc_entropies_and_stds_per_delta(self.features, l_set, curr_support, use_cosine_dist)
            values = np.asarray(values)
            max_degree = np.max(values)
            argmax = np.arange(len(values))[values == max_degree]
            delta = curr_support[argmax[-1]].item()
            print(f"DeltaScheduler | Max support size: {np.max(values)}, argmax delta: {round(delta, 5)}")
            logger.info(f"DeltaScheduler | Max support size: {np.max(values)}, argmax delta: {round(delta)}")
        elif policy == "max_deg_after_lt_2_take_min":
            assert current_max_deg is not None, "Current max degree must be set before using max_deg_after_lt_3_take_min policy."
            if current_max_deg >= 2:
                delta = self.current_delta
                print(f"DeltaScheduler | The current max degree is greater than or equal to 2. Keeping the current delta: {round(delta, 5)}")
                logger.info(f"DeltaScheduler | The current max degree is greater than or equal to 2. Keeping the current delta: {round(delta, 5)}")
            else:
                # if self.current_delta == np.min(curr_support):
                #     delta = self.current_delta
                #     print(f"DeltaScheduler | The current delta is the smallest in the support. Keeping the current delta: {round(delta, 5)}")
                #     logger.info(f"DeltaScheduler | The current delta is the smallest in the support. Keeping the current delta: {round(delta, 5)}")
                # else:
                _, _, values = calc_entropies_and_stds_per_delta(self.features, l_set, curr_support, use_cosine_dist)
                values = np.asarray(values)
                max_degree = np.max(values)
                argmax = np.arange(len(values))[values == max_degree]
                delta = curr_support[argmax[-1]].item()
                print(f"DeltaScheduler | Max support size: {np.max(values)}, argmax delta: {round(delta, 5)}")
                logger.info(f"DeltaScheduler | Max support size: {np.max(values)}, argmax delta: {round(delta)}")
        elif policy == "max_deg_after_lt_3_take_min":
            assert current_max_deg is not None, "Current max degree must be set before using max_deg_after_lt_3_take_min policy."
            if current_max_deg >= 3:
                delta = self.current_delta
                print(f"DeltaScheduler | The current max degree is greater than or equal to 3. Keeping the current delta: {round(delta, 5)}")
                logger.info(f"DeltaScheduler | The current max degree is greater than or equal to 3. Keeping the current delta: {round(delta, 5)}")
            else:
                # if self.current_delta == np.min(curr_support):
                #     delta = self.current_delta
                #     print(f"DeltaScheduler | The current delta is the smallest in the support. Keeping the current delta: {round(delta, 5)}")
                #     logger.info(f"DeltaScheduler | The current delta is the smallest in the support. Keeping the current delta: {round(delta, 5)}")
                # else:
                _, _, values = calc_entropies_and_stds_per_delta(self.features, l_set, curr_support, use_cosine_dist)
                values = np.asarray(values)
                max_degree = np.max(values)
                argmax = np.arange(len(values))[values == max_degree]
                delta = curr_support[argmax[-1]].item()
                print(f"DeltaScheduler | Max support size: {np.max(values)}, argmax delta: {round(delta, 5)}")
                logger.info(f"DeltaScheduler | Max support size: {np.max(values)}, argmax delta: {round(delta)}")
        elif policy == "max_deg_after_lt_4_take_min":
            assert current_max_deg is not None, "Current max degree must be set before using max_deg_after_lt_3_take_min policy."
            if current_max_deg >= 4:
                delta = self.current_delta
                print(f"DeltaScheduler | The current max degree is greater than or equal to 4. Keeping the current delta: {round(delta, 5)}")
                logger.info(f"DeltaScheduler | The current max degree is greater than or equal to 4. Keeping the current delta: {round(delta, 5)}")
            else:
                # if self.current_delta == np.min(curr_support):
                #     delta = self.current_delta
                #     print(f"DeltaScheduler | The current delta is the smallest in the support. Keeping the current delta: {round(delta, 5)}")
                #     logger.info(f"DeltaScheduler | The current delta is the smallest in the support. Keeping the current delta: {round(delta, 5)}")
                # else:
                _, _, values = calc_entropies_and_stds_per_delta(self.features, l_set, curr_support, use_cosine_dist)
                values = np.asarray(values)
                max_degree = np.max(values)
                argmax = np.arange(len(values))[values == max_degree]
                delta = curr_support[argmax[-1]].item()
                print(f"DeltaScheduler | Max support size: {np.max(values)}, argmax delta: {round(delta, 5)}")
                logger.info(f"DeltaScheduler | Max support size: {np.max(values)}, argmax delta: {round(delta)}")
        else:
            raise ValueError(f"Unsupported delta policy: {policy}")

        return delta, values, curr_support
