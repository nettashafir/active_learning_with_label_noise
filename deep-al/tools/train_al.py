import os
import sys
import traceback
import argparse
import numpy as np
import random
import torch
from copy import deepcopy
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(os.path.abspath('..'))
add_path(os.path.abspath('../pycls/al'))
add_path(os.path.abspath('../pycls/lnl'))
from pycls.al.ActiveLearning import ActiveLearning
from pycls.al.delta_scheduler import DeltaScheduler
from pycls.lnl.LearningWithNoisyLabels import LearningWithNoisyLabels
from pycls.lnl.lnl_utils import EnsembleNet
import pycls.core.builders as model_builder
from pycls.core.config import cfg, dump_cfg
import pycls.core.losses as losses
import pycls.core.optimizer as optim
from pycls.datasets.data import Data
from pycls.datasets.utils.helpers import create_subset
import pycls.utils.checkpoint as cu
import pycls.utils.logging as lu
import pycls.utils.metrics as mu
import pycls.utils.net as nu
from pycls.utils.meters import TestMeter
from pycls.utils.meters import TrainMeter
from tools.utils import seed_everything, get_budget, calc_metrics, print_table, save_stats

logger = lu.get_logger(__name__)

REPRESENTATION_BASED_AL = [
    "probcover", "prob_cover", "maxherding", "max_herding", "dcom",
    "npc", "probcover_nas", "weighted_npc", "maxherding_nas",
]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def argparser():
    parser = argparse.ArgumentParser(description='Active Learning - Image Classification')

    # General arguments
    parser.add_argument('--cfg', dest='cfg_file', help='Config file', required=True, type=str)
    parser.add_argument('--exp-name', help='Experiment Name', type=str, default='auto')
    parser.add_argument('--seed', help='Random seed', default=1, type=int)

    # Active Learning arguments
    parser.add_argument('--al', help='AL (Active Learning) Method', required=True, type=str)
    parser.add_argument('--initial_budget_size', help='Size of the initial random labeled set (for high budget AL methods)', default=0, type=int)
    parser.add_argument('--budget', help='Budget Per Round', type=int, default=0)
    parser.add_argument('--num_episodes', default=5, type=int)
    parser.add_argument("--samples_per_class", "--spc", help="Samples per Class per Round", nargs="+", type=int, default=[])

    # Active Learning arguments for Representation-Based Methods
    parser.add_argument('--features_normalization', help='Relevant only for Representation-Based AL Methods, e.g. ProbCover, MaxHerding and DCoM', default="unit_norm", type=str)
    parser.add_argument('--distance_metric', help='Relevant only for Graph-Based AL Methods, e.g. ProbCover and DCoM', default="l2", type=str)
    parser.add_argument('--initial_delta', help='Relevant only for ProbCover and DCoM', default=None, type=float)
    parser.add_argument('--delta_policy', help='Relevant only for ProbCover', default="constant", type=str)
    parser.add_argument('--delta_softening_policy', help='Relevant only for ProbCover', default=None, type=str)
    parser.add_argument('--use_noise_dropout', action="store_true", default=False)

    # Noise arguments
    parser.add_argument('--lnl', help='LNL (Learning with Noisy Labels) Method', type=str, default="ideal")
    parser.add_argument('--noise_type', type=str, help='sym, asym (for synthetic noise), noisy (for real-world datasets, default), aggre, worst, rand1, rand2, rand3 (for CIFAR10N)')
    parser.add_argument('--noise_rate', default=0.0, type=float, help='noise rate (for synthetic noise)')
    parser.add_argument('--train_on_all_labeled_data', action="store_true", default=False, help='Train on clean samples only')
    parser.add_argument('--noise_mom_coeff', default=0.0, type=float, help='the weight given for previous noise estimations')

    # Training arguments
    parser.add_argument('--use_linear_model', help='Whether to use a linear layer from self-supervised features', action='store_true', default=False)
    parser.add_argument('--use_1nn_model', help='Whether to use a 1NN from self-supervised features', action='store_true', default=False)
    parser.add_argument('--balance_classes', action='store_true', default=False)
    parser.add_argument('--k_logistic', default=50, type=int)
    parser.add_argument('--a_logistic', default=0.8, type=float)
    parser.add_argument('--max_epoch', help='Max Epoch', default=200, type=int)
    parser.add_argument('--finetune', help='Whether to continue with existing model between rounds', type=str2bool, default=False)

    # argument validations
    args = parser.parse_args()
    assert 0 <= args.noise_mom_coeff <= 1, 'Noise momentum should be in [0, 1]'
    assert args.budget > 0 or len(args.samples_per_class) > 0, 'Either budget or samples_per_class should be provided'

    return args


def main(cfg):
    # Getting the output directory ready (default is "/output")
    cfg.OUT_DIR = os.path.join(os.path.abspath('../..'), cfg.OUT_DIR)
    if not os.path.exists(cfg.OUT_DIR):
        os.mkdir(cfg.OUT_DIR)

    # Creating the experiment directory inside the dataset specific directory
    if cfg.EXP_NAME == 'auto':
        now = datetime.now()
        exp_dir = f'{now.year}_{now.month}_{now.day}_{now.hour:02}{now.minute:02}{now.second:02}_{now.microsecond}'
    else:
        exp_dir = cfg.EXP_NAME

    exp_dir = os.path.join(cfg.OUT_DIR, exp_dir)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)
        print("Experiment Directory is {}.\n".format(exp_dir))
        logger.info("Experiment Directory is {}.\n".format(exp_dir))
    else:
        print("Experiment Directory Already Exists: {}. Reusing it may lead to loss of old logs in the directory.\n".format(exp_dir))
        logger.info("Experiment Directory Already Exists: {}. Reusing it may lead to loss of old logs in the directory.\n".format(exp_dir))
    cfg.EXP_DIR = exp_dir

    # Save the config file in EXP_DIR
    dump_cfg(cfg)

    # Setup Logger
    lu.setup_logging(cfg)

    # Dataset preparing steps
    print("\n======== PREPARING DATA AND MODEL ========\n")
    logger.info("\n======== PREPARING DATA AND MODEL ========\n")
    if not os.path.exists(cfg.DATASET.ROOT_DIR):
        cfg.DATASET.ROOT_DIR = os.path.join(os.path.abspath('../..'), cfg.DATASET.ROOT_DIR)
    if cfg.NOISE.ROOT_NOISE_DIR is not None:
        cfg.NOISE.ROOT_NOISE_DIR = os.path.join(os.path.abspath('../..'), cfg.NOISE.ROOT_NOISE_DIR)
    data_obj = Data(cfg)
    project_features_before_training = cfg.MODEL.USE_1NN and cfg.ACTIVE_LEARNING.PROJECT_FEATURES_TO_UNIT_SPHERE
    only_features = cfg.MODEL.LINEAR_FROM_FEATURES or cfg.MODEL.USE_1NN
    train_data, train_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=True, isDownload=True, representation_model=cfg.DATASET.REPRESENTATION_MODEL, only_features=only_features, project_features=project_features_before_training)
    test_data, test_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=False, isDownload=True, representation_model=cfg.DATASET.REPRESENTATION_MODEL, only_features=only_features, project_features=project_features_before_training)
    print("\nDataset {} Loaded Sucessfully.\nTotal Train Size: {} and Total Test Size: {}\n".format(cfg.DATASET.NAME, train_size, test_size))
    logger.info("Dataset {} Loaded Sucessfully. Total Train Size: {} and Total Test Size: {}\n".format(cfg.DATASET.NAME, train_size, test_size))

    # Create a subset of the training data if necessary
    max_train_size = 100000  # Change if needed
    if cfg.ACTIVE_LEARNING.SAMPLING_FN in REPRESENTATION_BASED_AL and train_size > max_train_size:
        indices_path = f"{cfg.EXP_DIR}/{cfg.DATASET.NAME.lower()}_subset_{max_train_size}_indices_seed_{args.seed}.npy"
        if os.path.exists(indices_path):
            print("Loading the subset of the training data for faster computation.")
            logger.info("Loading the subset of the training data for faster computation.")
            indices = np.load(indices_path)
        else:
            print("Creating a subset of the training data for faster computation.")
            logger.info("Creating a subset of the training data for faster computation.")
            indices = np.random.choice(train_size, size=max_train_size, replace=False)
            np.save(indices_path, indices)
            print(f"Subset of the training data is saved at {indices_path}")
        train_data, train_size = create_subset(train_data, indices, cfg)

    # Initialize the labeled set (lSet) and unlabeled set (uSet)
    if cfg.ACTIVE_LEARNING.INITIAL_BUDGET_SIZE > 0:
        print(f"Creating random initial labeled set of size {cfg.ACTIVE_LEARNING.INITIAL_BUDGET_SIZE} from the training data.")
        logger.info(f"Creating random initial labeled set of size {cfg.ACTIVE_LEARNING.INITIAL_BUDGET_SIZE} from the training data.")
        indices = np.random.choice(train_size, size=cfg.ACTIVE_LEARNING.INITIAL_BUDGET_SIZE, replace=False)
        lSet = indices.astype(int)
        uSet = np.setdiff1d(np.arange(train_size), lSet).astype(int)
    else:
        print("Creating an empty labeled set and an unlabeled set containing all training data.")
        lSet, uSet = np.array([]), np.arange(train_size)  # Start with empty labeled set and all training data in unlabeled set


    # Initialize active learning (AL) and learning with noisy labels (LNL) objects
    lnl_obj = LearningWithNoisyLabels(train_data, test_data, cfg)
    test_loader = data_obj.getTestLoader(data=test_data, test_batch_size=cfg.TRAIN.BATCH_SIZE, seed_id=cfg.RNG_SEED)
    l_set_predicted_clean_indices = None
    run_DCoM = cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ['dcom']
    run_ProbCover = cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["prob_cover", "probcover", "npc", "weighted_npc"]
    delta_scheduler = DeltaScheduler(cfg, train_data.features) if run_ProbCover else None
    al_obj = ActiveLearning(data_obj, lnl_obj, cfg, delta_scheduler)

    # Initialize the model and optimizer
    if cfg.MODEL.USE_1NN:
        model = KNeighborsClassifier(n_neighbors=1)
        optimizer = model_init_state = opt_init_state = None
    else:
        model = model_builder.build_model(cfg)
        optimizer = optim.construct_optimizer(cfg, model)
        opt_init_state = deepcopy(optimizer.state_dict())
        model_init_state = deepcopy(model.state_dict().copy())
    checkpoint_file = None
    last_model = None

    print(f"Dataset: {cfg.DATASET.NAME}")
    print(f"Representation Model: {cfg.DATASET.REPRESENTATION_MODEL}")
    print(f"AL Query Method: {cfg.ACTIVE_LEARNING.SAMPLING_FN}")
    print(f"Max AL Episodes: {len(cfg.ACTIVE_LEARNING.BUDGET_PER_ROUND)}")
    print(f"Noise Type: {cfg.NOISE.NOISE_TYPE}")
    print(f"Noise rate: {cfg.NOISE.NOISE_RATE}")
    print(f"LNL algorithm: {cfg.NOISE.FILTERING_FN}")
    print(f"model: {cfg.MODEL.TYPE}")
    print(f"optimizer: {optimizer}")
    print(f"Class Balance: {cfg.TRAIN.BALANCE_CLASSES}")
    print(f"Max Epoch: {cfg.OPTIM.MAX_EPOCH}")

    logger.info(f"Dataset: {cfg.DATASET.NAME}")
    logger.info(f"Representation Model: {cfg.DATASET.REPRESENTATION_MODEL}")
    logger.info(f"AL Query Method: {cfg.ACTIVE_LEARNING.SAMPLING_FN}")
    logger.info(f"Max AL Episodes: {len(cfg.ACTIVE_LEARNING.BUDGET_PER_ROUND)}")
    logger.info(f"Noise Type: {cfg.NOISE.NOISE_TYPE}")
    logger.info(f"Noise rate: {cfg.NOISE.NOISE_RATE}")
    logger.info(f"LNL algorithm: {cfg.NOISE.FILTERING_FN}")
    logger.info(f"model: {cfg.MODEL.TYPE}")
    logger.info(f"optimizer: {optimizer}")
    logger.info(f"Class Balance: {cfg.TRAIN.BALANCE_CLASSES}")
    logger.info(f"Max Epoch: {cfg.OPTIM.MAX_EPOCH}")

    test_accuracies = []
    clean_true_positives = []
    clean_false_positives = []
    clean_false_negatives = []
    clean_true_negatives = []
    true_noise_ratios = []
    predicted_noise_ratios = []
    noise_accuracy_per_episode = []
    clean_recall_per_episode = []
    clean_precision_per_episode = []
    clean_f1_scores = []
    noise_recall_per_episode = []
    noise_precision_per_episode = []
    noise_f1_scores = []
    delta_avg_lst = []  # only for DCoM
    delta_std_lst = []  # only for DCoM
    delta_per_episode = []  # Only for ProbCover

    expected_clean_samples_per_class = cfg.ACTIVE_LEARNING.EXPECTED_CLEAN_SAMPLES_PER_CLASS
    cur_episode = 0
    # Start Active Learning cycle
    while cur_episode < cfg.ACTIVE_LEARNING.MAX_ITER:
        print("======== EPISODE {} BEGINS ========\n".format(cur_episode))
        logger.info("======== EPISODE {} BEGINS ========\n".format(cur_episode))

        # Creating output directory for the episode
        episode_dir = os.path.join(cfg.EXP_DIR, f'episode_{cur_episode}')
        if not os.path.exists(episode_dir):
            os.mkdir(episode_dir)
        cfg.EPISODE_DIR = episode_dir

        # --------------------------- Query Selection (+ Annotation) ---------------------------
        print("\n======== ACTIVE SAMPLING ========\n")
        logger.info("\n======== ACTIVE SAMPLING ========\n")
        curr_budget_size = cfg.ACTIVE_LEARNING.BUDGET_PER_ROUND[cur_episode]
        cfg.ACTIVE_LEARNING.BUDGET_SIZE = curr_budget_size

        if run_DCoM:
            if len(lSet) == 0:
                print('Labeled Set is Empty - Create and save the first delta values list')
                logger.info('Labeled Set is Empty - Create and save the first delta values list')
                lSet_deltas = [str(cfg.ACTIVE_LEARNING.INITIAL_DELTA)] * cfg.ACTIVE_LEARNING.BUDGET_PER_ROUND[0]
                cfg.ACTIVE_LEARNING.DELTA_LST = lSet_deltas
                delta_avg_lst.append(cfg.ACTIVE_LEARNING.INITIAL_DELTA)
                delta_std_lst.append(0)
            else:
                # DCoM's delta-s updating
                print("\n======== Update the deltas dynamically ========\n")
                logger.info("\n======== Update the deltas dynamically ========\n")
                from pycls.al.DCoM import DCoM
                last_budget_size = cfg.ACTIVE_LEARNING.BUDGET_PER_ROUND[cur_episode-1]
                al_algo = DCoM(cfg, lSet, uSet, budgetSize=last_budget_size,
                               max_delta=cfg.ACTIVE_LEARNING.MAX_DELTA,
                               lSet_deltas=cfg.ACTIVE_LEARNING.DELTA_LST)

                lSet_labels = np.take(train_data.targets, np.asarray(lSet, dtype=np.int64))
                all_images_idx = np.array(list(lSet) + list(uSet))
                images_loader = data_obj.getSequentialDataLoader(indexes=all_images_idx, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
                all_labels = np.take(train_data.targets, np.asarray(all_images_idx, dtype=np.int64))
                images_pseudo_labels = get_label_from_model(images_loader, checkpoint_file, cfg, last_model)
                cfg.ACTIVE_LEARNING.DELTA_LST[-1 * last_budget_size:] \
                    = al_algo.new_centroids_deltas(lSet_labels,
                                                   all_labels=all_labels,
                                                   pseudo_labels=images_pseudo_labels,
                                                   budget=last_budget_size)

                delta_lst_float = [float(delta) for delta in cfg.ACTIVE_LEARNING.DELTA_LST]
                delta_avg_lst.append(np.average(delta_lst_float))
                delta_std_lst.append(np.std(delta_lst_float))

        activeSet, new_uSet = al_obj.sample_from_uSet(model, lSet, uSet, train_data, budget_size=curr_budget_size, lSetCleanIdx=l_set_predicted_clean_indices)
        lSet = np.append(lSet, activeSet).astype(int)  # Add activeSet to lSet, save new_uSet as uSet and update dataloader for the next episode
        uSet = new_uSet.astype(int)
        if run_ProbCover:
            delta_per_episode.append(delta_scheduler.current_delta)

        print("Active Sampling Completed. After Episode {}, SPC {}:\nNew Labeled Set: {}, New Unlabeled Set: {}, Active Set: {}\n".format(cur_episode, expected_clean_samples_per_class[cur_episode], len(lSet), len(uSet), len(activeSet)))
        logger.info("Active Sampling Completed. After Episode {}, SPC {}:\nNew Labeled Set: {}, New Unlabeled Set: {}, Active Set: {}\n".format(cur_episode, expected_clean_samples_per_class[cur_episode], len(lSet), len(uSet), len(activeSet)))

        # add avg delta to cfg.ACTIVE_LEARNING.DELTA_LST towards the next active sampling
        if run_DCoM:
            delta_lst_float = [float(delta) for delta in cfg.ACTIVE_LEARNING.DELTA_LST]
            next_initial_deltas = [str(round(np.average(delta_lst_float), 2))] * curr_budget_size
            cfg.ACTIVE_LEARNING.DELTA_LST.extend(next_initial_deltas)
            print("Current delta list: ", cfg.ACTIVE_LEARNING.DELTA_LST)
            print("Current delta avg list: ", delta_avg_lst)
            print("Current delta std list: ", delta_std_lst)

            logger.info("Current delta list: {}\n".format(cfg.ACTIVE_LEARNING.DELTA_LST))
            logger.info("Current delta avg list: {}\n".format(delta_avg_lst))
            logger.info("Current delta std list: {}\n".format(delta_std_lst))


        # --------------------------- Noise Filtering ---------------------------
        print("======== IDENTIFY NOISY SAMPLES ========\n")
        logger.info("======== IDENTIFY NOISY SAMPLES ========\n")
        if hasattr(train_data, 'is_noisy'):
            l_set_true_clean_indices = np.logical_not(np.take(train_data.is_noisy, lSet.astype(int)))
        else:
            l_set_true_clean_indices = np.take(train_data.noisy_labels, lSet.astype(int)) == np.take(train_data.targets, lSet.astype(int))
        l_set_predicted_clean_indices, scores, confidence_scores, lnl_model, lnl_train_loss = lnl_obj.identify_noisy_samples(lSet, uSet, return_scores=True)

        # calculate noise filtering metrics
        clean_true_positive = np.sum(np.logical_and(l_set_true_clean_indices == 1, l_set_predicted_clean_indices == 1))
        clean_false_positive = np.sum(np.logical_and(l_set_true_clean_indices == 0, l_set_predicted_clean_indices == 1))
        clean_false_negative = np.sum(np.logical_and(l_set_true_clean_indices == 1, l_set_predicted_clean_indices == 0))
        clean_true_negative = np.sum(np.logical_and(l_set_true_clean_indices == 0, l_set_predicted_clean_indices == 0))

        clean_true_positives.append(clean_true_positive)
        clean_false_positives.append(clean_false_positive)
        clean_false_negatives.append(clean_false_negative)
        clean_true_negatives.append(clean_true_negative)

        accuracy, clean_precision, clean_recall, clean_f1, noise_precision, noise_recall, noise_f1, true_noise, predicted_noise = \
            calc_metrics(clean_true_positive, clean_false_positive, clean_true_negative, clean_false_negative)
        noise_accuracy_per_episode.append(accuracy)
        true_noise_ratios.append(round(true_noise, 3))
        predicted_noise_ratios.append(round(predicted_noise, 3))
        clean_recall_per_episode.append(clean_recall)
        clean_precision_per_episode.append(clean_precision)
        clean_f1_scores.append(clean_f1)
        noise_recall_per_episode.append(noise_recall)
        noise_precision_per_episode.append(noise_precision)
        noise_f1_scores.append(noise_f1)


        # --------------------------- Model Training ---------------------------
        if checkpoint_file is not None and os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

        print("\n======== TRAINING ========")
        logger.info("\n======== TRAINING ========")

        if args.train_on_all_labeled_data:
            print("Training on all labeled data")
            train_set = lSet
        else:
            print("Training only on clean samples")
            train_set = lSet[l_set_predicted_clean_indices]

        if cfg.TRAIN.BALANCE_CLASSES:
            lSet_loader = data_obj.getBalancedIndexesDataLoader(indexes=train_set, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
        else:
            lSet_loader = data_obj.getIndexesDataLoader(indexes=train_set, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)

        train_loss = train_model(lSet_loader, model, optimizer, cfg, train_set)

        # Save the model
        if not cfg.MODEL.USE_1NN:
            model.eval()
            last_model = model
            last_model_state = model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict()
            checkpoint_file = cu.save_checkpoint(info="train_loss_" + str(round(train_loss, 4)), model_state=last_model_state, optimizer_state=optimizer.state_dict(), epoch=cfg.OPTIM.MAX_EPOCH, cfg=cfg)
            print('\nWrote Last Model Checkpoint to: {}\n'.format(checkpoint_file.split('/')[-1]))
            logger.info('\nWrote Last Model Checkpoint to: {}\n'.format(checkpoint_file))


        # --------------------------- Model Evaluation ---------------------------
        print("\n======== TESTING ========\n")
        logger.info("\n======== TESTING ========\n")
        test_acc = test_model(test_loader, cfg, cur_episode, model, checkpoint_file)
        test_accuracies.append(test_acc)

        # print episode stats
        print("EPISODE {} Test Accuracy: {}.\n".format(cur_episode, round(test_acc, 4)))
        logger.info("EPISODE {} Test Accuracy {}.\n".format(cur_episode, test_acc))

        headers = [f"{spc} SPC" for spc in expected_clean_samples_per_class[:cur_episode + 1]]
        data_for_table = [np.asarray(test_accuracies).round(3), np.concatenate(([cfg.ACTIVE_LEARNING.BUDGET_PER_ROUND[0] + cfg.ACTIVE_LEARNING.INITIAL_BUDGET_SIZE], np.cumsum(cfg.ACTIVE_LEARNING.BUDGET_PER_ROUND)[1:cur_episode+1]))]
        rows = ["Test Accuracy", "Total Budgets"]
        if cfg.NOISE.NOISE_RATE > 0:
            extra_rows = [np.asarray(true_noise_ratios).round(3)]
            extra_rows_names = ["True Noise Ratios"]
            if cfg.NOISE.FILTERING_FN != "ideal":
                extra_rows += [np.asarray(predicted_noise_ratios).round(3),
                                    np.asarray(noise_accuracy_per_episode).round(3),
                                    np.asarray(clean_recall_per_episode).round(3),
                                    np.asarray(clean_precision_per_episode).round(3),
                                    np.asarray(clean_f1_scores).round(3)]
                extra_rows_names += ["Predicted Noise Ratios",
                                     "Noise Accuracy",
                                     "Clean Recall",
                                     "Clean Precision",
                                     "Clean F1"]
            data_for_table = extra_rows + data_for_table
            rows = extra_rows_names + rows
        if run_DCoM:
            data_for_table.append(np.asarray(delta_avg_lst).round(4))
            data_for_table.append(np.asarray(delta_std_lst).round(4))
            rows.append("Delta Avg")
            rows.append("Delta Std")
        if run_ProbCover:
            data_for_table.append(np.asarray(delta_per_episode).round(5))
            rows.append("ProbCover Delta")
        print(f"time - {str(datetime.now())}\n")
        logger.info(f"time - {str(datetime.now())}\n")
        try:
            print_table(data_for_table, headers, rows, logger=logger)
        except IndexError as e:
            print(f"Error printing table: {e}")
            logger.error(f"Error printing table: {e}")
            print("Headers: ", headers)
            logger.info("Headers: ", headers)
            for i in range(len(data_for_table)):
                print(f"{rows[i]}: {data_for_table[i]}")
                logger.info(f"{rows[i]}: {data_for_table[i]}")

        # Save current lSet, new_uSet and activeSet in the episode directory
        print("Saving the current episode data in the episode directory")
        data_obj.saveSets(lSet, uSet, activeSet, cfg.EPISODE_DIR)
        lnl_obj.save_checkpoint(cfg.EPISODE_DIR, lSet)
        save_stats(cfg.EPISODE_DIR, test_accuracies, clean_true_positives, clean_false_positives, clean_false_negatives, clean_true_negatives)
        if run_DCoM:
            np.save(os.path.join(cfg.EPISODE_DIR, 'delta_list.npy'), cfg.ACTIVE_LEARNING.DELTA_LST)
            np.save(os.path.join(cfg.EPISODE_DIR, 'avg_delta_per_episode_list.npy'), delta_avg_lst)
            np.save(os.path.join(cfg.EPISODE_DIR, 'std_delta_per_episode_list.npy'), delta_std_lst)
        elif run_ProbCover:
            np.save(os.path.join(cfg.EPISODE_DIR, 'delta_per_episode_list.npy'), delta_per_episode)

        cur_episode += 1

        print("\n================================\n\n")
        logger.info("\n================================\n\n")

        if not cfg.ACTIVE_LEARNING.FINE_TUNE:
            # start model from scratch
            print('Starting model from scratch - ignoring existing weights.')
            logger.info('Starting model from scratch - ignoring existing weights.')

            if cfg.MODEL.USE_1NN:
                model = KNeighborsClassifier(n_neighbors=1)
                optimizer = model_init_state = opt_init_state = None
            else:
                model = model_builder.build_model(cfg)
                optimizer = optim.construct_optimizer(cfg, model)
                print(model.load_state_dict(model_init_state))
                print(optimizer.load_state_dict(opt_init_state))

    save_stats(cfg.EXP_DIR, test_accuracies, clean_true_positives, clean_false_positives, clean_false_negatives, clean_true_negatives)
    print("Saved the final accuracies and metrics")
    logger.info("Saved the final accuracies and metrics")


def train_model(train_loader, model, optimizer, cfg, lSet):
    if cfg.MODEL.USE_1NN:
        data = np.take(train_loader.dataset.features, lSet, axis=0)
        labels = np.take(train_loader.dataset.noisy_labels, lSet)
        model.fit(data, labels)
        train_loss = 0

    else:
        start_epoch = 0
        loss_fun = losses.get_loss_fun()

        # Create meters
        train_meter = TrainMeter(len(train_loader))

        clf_train_iterations = cfg.OPTIM.MAX_EPOCH * int(len(train_loader) / cfg.TRAIN.BATCH_SIZE)
        clf_change_lr_iter = clf_train_iterations // 25
        clf_iter_count = 0
        train_loss = np.inf

        for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):

            # Train for one epoch
            train_loss, clf_iter_count = train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch,
                                                     cfg, clf_iter_count, clf_change_lr_iter, clf_train_iterations)

            # Compute precise BN stats
            if cfg.BN.USE_PRECISE_STATS:
                nu.compute_precise_bn_stats(model, train_loader)

            if cur_epoch == 0 or (cur_epoch + 1) % 25 == 0:
                print('Training Epoch: {}/{}\tTrain Loss: {}\t'.format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH, round(train_loss, 4)))
                logger.info('Training Epoch: {}/{}\tTrain Loss: {}\t'.format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH, round(train_loss, 4)))

    return train_loss


def test_model(test_loader, cfg, cur_episode, model, checkpoint_file=None):

    if cfg.MODEL.USE_1NN:
        predictions = model.predict(test_loader.dataset.features)
        test_acc = 100. * (predictions == np.array(test_loader.dataset.targets)).mean()
    else:

        test_meter = TestMeter(len(test_loader))

        # model = model_builder.build_model(cfg)
        # model = cu.load_checkpoint(checkpoint_file, model)

        if isinstance(model, EnsembleNet):
            model.cuda()
            model.eval()
            predictions = np.zeros(len(test_loader.dataset))
            for cur_iter, (inputs, labels, indices) in enumerate(test_loader):
                inputs = inputs.cuda()
                preds = model(inputs)
                predictions[indices] = preds.cpu().numpy()
            test_acc = 100. * (predictions == np.array(test_loader.dataset.targets)).mean()
        else:
            test_err = test_epoch(test_loader, model, test_meter, cur_episode)
            test_acc = 100. - test_err

    return test_acc


def train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch, cfg, clf_iter_count, clf_change_lr_iter, clf_max_iter):
    """Performs one epoch of training."""

    # Shuffle the data
    # loader.shuffle(train_loader, cur_epoch)
    if cfg.NUM_GPUS > 1:
        train_loader.sampler.set_epoch(cur_epoch)

    # Update the learning rate
    # Currently we only support LR schedules for only 'SGD' optimizer
    lr = optim.get_epoch_lr(cfg, cur_epoch)
    if cfg.OPTIM.TYPE == "sgd":
        optim.set_lr(optimizer, lr)

    if torch.cuda.is_available():
        model.cuda()

    # Enable training mode
    model.train()

    loss = None
    for cur_iter, batch in enumerate(train_loader):
        inputs, noisy_labels, _, indices = batch
        labels = noisy_labels  # if args.noise_type == clean, then noisy_labels are actually clean labels
        #ensuring that inputs are floatTensor as model weights are
        inputs = inputs.type(torch.cuda.FloatTensor)
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Perform the forward pass
        preds = model(inputs)
        # Compute the loss
        loss = loss_fun(preds, labels)
        # Perform the backward pass
        optimizer.zero_grad()
        loss.backward()
        # Update the parametersSWA
        optimizer.step()
        # Compute the errors
        top1_err, top5_err = mu.topk_errors(preds, labels, [1, 5])
        # Copy the stats from GPU to CPU (sync point)
        loss, top1_err = loss.item(), top1_err.item()

    return loss, clf_iter_count


def get_label_from_model(images_loader, checkpoint_file, cfg, model=None):
    """
    returns the labels of the images according to the checkpoint file model
    """
    get_label_meter = TestMeter(len(images_loader))
    if model is None:
        model = model_builder.build_model(cfg)
        model = cu.load_checkpoint(checkpoint_file, model)

    pred = get_label_epoch(images_loader, model, get_label_meter)
    return pred


@torch.no_grad()
def test_epoch(test_loader, model, test_meter, cur_epoch):
    """Evaluates the model on the test set."""

    if torch.cuda.is_available():
        model.cuda()

    # Enable eval mode
    model.eval()

    misclassifications = 0.
    totalSamples = 0.

    for cur_iter, batch in enumerate(test_loader):
        if test_loader.dataset.train:  # for cross-validation
            inputs, noisy_labels, _, indices = batch
            labels = noisy_labels  # if args.noise_type == clean, then noisy_labels are actually clean labels
        else:
            inputs, labels, indices = batch
        with torch.no_grad():
            # Transfer the data to the current GPU device
            inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
            inputs = inputs.type(torch.cuda.FloatTensor)
            # Compute the predictions
            preds = model(inputs)
            # Compute the errors
            top1_err, top5_err = mu.topk_errors(preds, labels, [1, 5])
            # Copy the errors from GPU to CPU (sync point)
            top1_err = top1_err.item()
            # Multiply by Number of GPU's as top1_err is scaled by 1/Num_GPUs
            misclassifications += top1_err * inputs.size(0) * cfg.NUM_GPUS
            totalSamples += inputs.size(0)*cfg.NUM_GPUS
            test_meter.iter_toc()
            # Update and log stats
            test_meter.update_stats(
                top1_err=top1_err, mb_size=inputs.size(0) * cfg.NUM_GPUS
            )

    return misclassifications/totalSamples


@torch.no_grad()
def get_label_epoch(images_loader, model, get_label_meter):
    """get labels according to the model."""
    if torch.cuda.is_available():
        model.cuda()

    # Enable eval mode
    model.eval()
    get_label_meter.iter_tic()

    all_preds = []
    for cur_iter, (inputs, _, _, _) in enumerate(images_loader):
        with torch.no_grad():
            # Transfer the data to the current GPU device
            inputs = inputs.cuda().type(torch.cuda.FloatTensor)
            # Compute the predictions
            preds = model(inputs)
            all_preds += preds

    final_preds = [torch.argmax(p).item() for p in all_preds]
    model.train()

    return final_preds


if __name__ == "__main__":
    args = argparser()
    print(args)

    cfg.merge_from_file(args.cfg_file)
    cfg.RNG_SEED = args.seed
    seed_everything(cfg.RNG_SEED)
    cfg.EXP_NAME = args.exp_name

    # Active Learning
    cfg.ACTIVE_LEARNING.SAMPLING_FN = args.al
    cfg.ACTIVE_LEARNING.A_LOGISTIC = args.a_logistic
    cfg.ACTIVE_LEARNING.K_LOGISTIC = args.k_logistic
    cfg.ACTIVE_LEARNING.DELTA_LST = []  # relevant for DCoM only
    cfg.ACTIVE_LEARNING.NOISE_DROPOUT = args.use_noise_dropout
    cfg.ACTIVE_LEARNING.USE_COSINE_DIST = (args.distance_metric == "cosine")

    # Active Learning - delta hyperparameter (relevant for ProbCover, DCoM and MaxHerding)
    cfg.ACTIVE_LEARNING.INITIAL_DELTA = args.initial_delta  # for MaxHerding this parameter is sigma
    cfg.ACTIVE_LEARNING.PROJECT_FEATURES_TO_UNIT_SPHERE = (args.features_normalization == "unit_norm")
    cfg.ACTIVE_LEARNING.DELTA_POLICY.MAJOR_POLICY = args.delta_policy
    cfg.ACTIVE_LEARNING.DELTA_POLICY.SOFTENING_POLICY = args.delta_softening_policy
    cfg.ACTIVE_LEARNING.MAX_DELTA = 0.5 if cfg.ACTIVE_LEARNING.USE_COSINE_DIST else 1.1
    cfg.ACTIVE_LEARNING.DELTA_RESOLUTION = 0.02

    # Noisy Labels
    cfg.NOISE.FILTERING_FN = args.lnl
    cfg.NOISE.NOISE_TYPE = cfg.NOISE.NOISE_TYPE or args.noise_type
    assert cfg.NOISE.NOISE_TYPE is not None, 'Noise type must be specified. Use --noise_type argument, or add it to the config file.'
    valid_noise_types = [
        "sym", "asym",  # Synthetic Noise
        "noisy_label",  # Instance Dependant Noise - For CIFAR100N, Clothing1M, Food101N, WebVision
        "worse_label", "aggre_label", "random_label1", "random_label2", "random_label3", "clean_label"  # Instance Dependant Noise for CIFAR10N only
    ]
    assert cfg.NOISE.NOISE_TYPE in valid_noise_types, f'Invalid noise type. Pick from: {valid_noise_types}'
    assert not (cfg.DATASET.NAME in ["CIFAR10N", "CIFAR100N", "CLOTHING1M", "FOOD101N", "MINIWEBVISION_FLICKR", "MINIWEBVISION_GOOGLE"] and cfg.NOISE.NOISE_TYPE != "noisy_label"), "For CLOTHING1M, FOOD101N and MINIWEBVISION datasets, the noise type must be 'noisy_label'."
    assert not (cfg.DATASET.NAME in ["CIFAR10N", "CIFAR100N", "CLOTHING1M", "FOOD101N", "MINIWEBVISION_FLICKR", "MINIWEBVISION_GOOGLE"] and args.noise_rate != 0.0), "Custom noise rate is not applicable for real-world noisy datasets."
    if cfg.NOISE.NOISE_TYPE in ['sym', 'asym']:
        cfg.NOISE.NOISE_RATE = args.noise_rate
    assert cfg.NOISE.NOISE_RATE is not None, 'Noise rate must be specified. Use --noise_rate argument (when using synthetic noise), or add it to the config file.'
    cfg.NOISE.MOMENTUM_COEFFICIENT = args.noise_mom_coeff

    # Active Learning - Budget
    cfg.ACTIVE_LEARNING.INITIAL_BUDGET_SIZE = args.initial_budget_size
    if args.budget > 0:
        cfg.ACTIVE_LEARNING.BUDGET_SIZE = args.budget
        budget_per_round = [args.budget] * args.num_episodes
        cfg.ACTIVE_LEARNING.EXPECTED_CLEAN_SAMPLES_PER_CLASS = (np.cumsum(budget_per_round) / cfg.MODEL.NUM_CLASSES) * (1 - cfg.NOISE.NOISE_RATE)
    elif len(args.samples_per_class) > 0:
        cfg.ACTIVE_LEARNING.EXPECTED_CLEAN_SAMPLES_PER_CLASS = args.samples_per_class
        budgets = [get_budget(cfg.NOISE.NOISE_RATE, spc, cfg.MODEL.NUM_CLASSES) for spc in args.samples_per_class]
        budget_per_round = [budgets[0]] + [budgets[i] - budgets[i-1] for i in range(1, len(budgets))]
    else:
        raise ValueError('Either budget or samples_per_class should be provided')
    cfg.ACTIVE_LEARNING.MAX_ITER = len(budget_per_round)
    cfg.ACTIVE_LEARNING.BUDGET_PER_ROUND = budget_per_round
    cfg.ACTIVE_LEARNING.BUDGET_SIZE = budget_per_round[0]

    # Model & Training
    cfg.MODEL.USE_1NN = args.use_1nn_model
    if cfg.MODEL.USE_1NN:
        cfg.MODEL.TYPE = '1NN'
    cfg.MODEL.LINEAR_FROM_FEATURES = args.use_linear_model
    if cfg.MODEL.LINEAR_FROM_FEATURES:
        cfg.MODEL.TYPE = 'linear'
    cfg.OPTIM.MAX_EPOCH = args.max_epoch
    cfg.TRAIN.BALANCE_CLASSES = cfg.TRAIN.BALANCE_CLASSES or args.balance_classes
    cfg.ACTIVE_LEARNING.FINE_TUNE = args.finetune

    main(cfg)
