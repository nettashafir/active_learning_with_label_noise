import os
import sys
import traceback
import argparse
import numpy as np
import random
import torch
from torch.utils.data import Dataset, Subset
from copy import deepcopy
from sklearn.neighbors import KNeighborsClassifier
import warnings
from datetime import datetime
# import wandb # TODO
# wandb.init(config=config, project='noisylabel', entity='goguryeo', name='_'.join(wandb_run_name_list))

os.environ["HOME"] = "/cs/labs/daphna/nettashaf"
os.environ['TORCH_HOME'] = '/cs/labs/daphna/nettashaf/.cache'

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


add_path(os.path.abspath('..'))
add_path(os.path.abspath('../pycls/al'))
add_path(os.path.abspath('../pycls/lnl'))
from pycls.al.ActiveLearning import ActiveLearning, calculate_coverage, choose_delta_for_probcover, set_initial_delta
from pycls.al.delta_scheduler import DeltaScheduler
from pycls.lnl.LearningWithNoisyLabels import LearningWithNoisyLabels
from pycls.lnl.lnl_utils import EnsembleNet
import pycls.core.builders as model_builder
from pycls.core.config import cfg, dump_cfg
import pycls.core.losses as losses
import pycls.core.optimizer as optim
from pycls.datasets.data import Data
import pycls.utils.checkpoint as cu
import pycls.utils.logging as lu
import pycls.utils.metrics as mu
import pycls.utils.net as nu
from pycls.utils.meters import TestMeter
from pycls.utils.meters import TrainMeter
from pycls.datasets.utils.features import extract_features_from_model, set_new_features
from tools.utils import calc_metrics, print_table

logger = lu.get_logger(__name__)

plot_episode_xvalues = []
plot_episode_yvalues = []

plot_epoch_xvalues = []
plot_epoch_yvalues = []

plot_it_x_values = []
plot_it_y_values = []


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_budget(noise_ratio, samples_per_class, num_classes):
  budget = int(np.ceil((samples_per_class * num_classes) / (1-noise_ratio)))
  return budget


def save_stats(path, test_accuracies, clean_true_positives, clean_false_positives, clean_false_negatives, clean_true_negatives):
    np.save(os.path.join(path, 'test_accuracies.npy'), test_accuracies)
    np.save(os.path.join(path, 'clean_true_positives.npy'), clean_true_positives)
    np.save(os.path.join(path, 'clean_true_negatives.npy'), clean_true_negatives)
    np.save(os.path.join(path, 'clean_false_positives.npy'), clean_false_positives)
    np.save(os.path.join(path, 'clean_false_negatives.npy'), clean_false_negatives)


def argparser():
    parser = argparse.ArgumentParser(description='Active Learning - Image Classification')

    # General arguments
    parser.add_argument('--cfg', dest='cfg_file', help='Config file', required=True, type=str)
    parser.add_argument('--rep', help='Representation Learning Model', type=str, default=None)
    parser.add_argument('--exp-name', help='Experiment Name', type=str, default='auto')
    parser.add_argument('--seed', help='Random seed', default=1, type=int)
    parser.add_argument('--force_run', action="store_true", default=False)
    parser.add_argument('--debug', action="store_true", default=False)

    # Active Learning arguments
    parser.add_argument('--al', help='AL (Active Learning) Method', required=True, type=str)
    parser.add_argument('--change_al_to', help='Second AL (Active Learning) Method', default=None, type=str)
    parser.add_argument('--change_al_after', default=None, type=int)

    parser.add_argument('--cosine_dist', action="store_true", default=False)
    parser.add_argument('--cr_delta', help='Relevant only for ProbCover and DCoM', default=0.65, type=float)
    parser.add_argument('--update_features_in_cr', action='store_true', default=False)

    parser.add_argument('--initial_delta', help='Relevant only for ProbCover and DCoM', default=None, type=float)
    parser.add_argument('--delta_policy', required=False, default="constant", type=str)
    parser.add_argument('--delta_softening_policy', required=False, default=None, type=str)
    parser.add_argument('--delta_consider_noise', action='store_true', default=False)

    parser.add_argument('--softmax_temp', help='Relevant only for ProbCoverSampling', default=1, type=float)
    parser.add_argument('--softmax_temp_policy', help='Relevant only for ProbCoverSampling', default="constant", type=str)
    parser.add_argument('--softmax_temp_policy_a', help='Relevant only for ProbCoverSampling', default=1.0, type=float)
    parser.add_argument('--softmax_temp_policy_b', help='Relevant only for ProbCoverSampling', default=0.0, type=float)
    parser.add_argument('--softmax_temp_policy_k', help='Relevant only for ProbCoverSampling', default=10, type=float)

    parser.add_argument('--greedy_selection', action='store_true', default=False)
    parser.add_argument('--super_greedy_selection', action='store_true', default=False)
    parser.add_argument('--noisy_label_inference', action='store_true', default=False)
    parser.add_argument('--use_noise_dropout', action='store_true', default=False)

    # budget arguments
    parser.add_argument('--l_set_path', default=None, type=str)
    parser.add_argument('--initial_size', help='Size of the initial random labeled set', default=0, type=int)
    parser.add_argument('--initial_budget', help='Initial budget for the first query selection', default=0, type=int)
    parser.add_argument('--budget', help='Budget Per Round', type=int, default=0)
    parser.add_argument('--num_episodes', default=5, type=int)
    parser.add_argument("--samples_per_class", help="Samples per Class per Round", nargs="+", type=int, default=[])
    parser.add_argument("--cumulative_budget", nargs="+", type=int, default=[])

    # Noise arguments
    parser.add_argument('--lnl', help='LNL (Learning with Noisy Labels) Method', required=True, type=str)
    parser.add_argument('--noise_type', type=str, help='clean, sym, asym, aggre, worst, rand1, rand2, rand3', default='clean')
    parser.add_argument('--noise_rate', default=0.0, type=float, help='noise rate for synthetic noise')
    parser.add_argument('--train_on_all_labeled_data', action="store_true", default=False, help='Train on clean samples only')
    parser.add_argument('--train_on_most_confident', action="store_true", default=False, help='Train on constant amount of samples')
    parser.add_argument('--noise_mom_coeff', default=0.0, type=float, help='the weight given for previous noise estimations')
    parser.add_argument('--use_neighbors_for_threshold', action="store_true", default=False)

    # Training arguments
    # parser.add_argument('--finetune', help='Whether to continue with existing model between rounds', type=str2bool, default=False)
    parser.add_argument('--linear_from_features', help='Whether to use a linear layer from self-supervised features', action='store_true', default=False)
    parser.add_argument('--use_1nn', action='store_true', default=False)
    parser.add_argument('--use_lnl_model', action='store_true', default=False)
    parser.add_argument('--balance_classes', action='store_true', default=False)
    parser.add_argument('--k_logistic', default=50, type=int)
    parser.add_argument('--a_logistic', default=0.8, type=float)
    parser.add_argument('--max-epoch', help='Max Epoch', default=200, type=int)

    # argument validations
    args = parser.parse_args()
    if args.use_1nn:
        args.linear_from_features = True
    if args.delta_softening_policy is not None and args.delta_policy in ["constant", "linear_descent", "exponential_descent"]:
        print(f"Delta softening policy is not applicable for {args.delta_policy} delta policy. Ignoring delta softening policy.")
        args.delta_softening_policy = None
    assert not (args.change_al_to is not None and args.change_al_after is None), "Change AL after must be provided"
    assert not (args.change_al_to is None and args.change_al_after is not None), "Change AL to must be provided"
    assert not (args.softmax_temp_policy in ["linear", "exponential"] and args.softmax_temp_policy_b == 0.0), "Softmax temperature policy a cannot be 0"
    assert not (args.delta_consider_noise is True and args.noise_rate == 0.0), "Cannot consider noise when noise rate is 0"
    assert not (args.update_features_in_cr and args.linear_from_features), "Cannot update features in CR and use linear layer from features"
    assert not ((args.greedy_selection or args.super_greedy_selection) and "noisy" not in args.al), "Greedy selection is only for noisy oracle AL methods"
    assert not (args.use_noise_dropout and "noisy_oracle" not in args.al), "Noise dropout is only for probcover noisy oracle"
    assert not (args.use_neighbors_for_threshold and args.lnl != "aum"), "Neighbors are only for AUM"
    assert not (args.noisy_label_inference and not (args.greedy_selection or args.super_greedy_selection)), "Noisy label inference is only for greedy selection"
    assert not (args.train_on_all_labeled_data and args.train_on_most_confident), "Cannot train on all labeled data and most confident"
    if args.noisy_label_inference and args.greedy_selection:
        warnings.warn("Noisy label inference is only for super greedy selection. Setting super greedy selection to True")
        logger.warning("Noisy label inference is only for super greedy selection. Setting super greedy selection to True")
    assert not (args.linear_from_features and args.lnl in ['unicon']), "SSL algorithms must use the original images"
    noise_type_map = {'sym': 'sym', 'asym': 'asym',
                     'clean': 'clean_label',  'worse': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1',
                      'rand2': 'random_label2', 'rand3': 'random_label3',
                      "noisy_label": "noisy_label"
                      }
    assert args.noise_type in noise_type_map.keys(), f'Invalid noise type. Pick from: {list(noise_type_map.keys())}'
    args.noise_type = noise_type_map[args.noise_type]
    assert 0 <= args.noise_mom_coeff <= 1, 'Noise momentum should be in [0, 1]'
    assert args.budget > 0 or len(args.samples_per_class) > 0 or len(args.cumulative_budget) > 0, 'Either budget or samples_per_class should be provided'

    return args


def seed_everything(seed):
    random.seed(seed)  # Python random module.
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def main(cfg):
    global plot_episode_yvalues

    # Setting up GPU args
    use_cuda = (cfg.NUM_GPUS > 0) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': cfg.DATA_LOADER.NUM_WORKERS, 'pin_memory': cfg.DATA_LOADER.PIN_MEMORY} if use_cuda else {}

    # Auto assign a RNG_SEED when not supplied a value
    if cfg.RNG_SEED is None:
        cfg.RNG_SEED = np.random.randint(100)

    # Using specific GPU
    # os.environ['NVIDIA_VISIBLE_DEVICES'] = str(cfg.GPU_ID)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # print("Using GPU : {}.\n".format(cfg.GPU_ID))

    # Getting the output directory ready (default is "/output")
    cfg.OUT_DIR = "/cs/labs/daphna/nettashaf/TypiClustNoisy/output/"
    if not os.path.exists(cfg.OUT_DIR):
        os.mkdir(cfg.OUT_DIR)

    # Creating the experiment directory inside the dataset specific directory
    # all logs, labeled, unlabeled, validation sets are stroed here
    # E.g., output/CIFAR10/resnet18/{timestamp or cfg.EXP_NAME based on arguments passed}
    al_name = args.al
    if cfg.ACTIVE_LEARNING.SUPER_GREEDY_SELECTION:
        al_name = f"super_greedy_{al_name}"
    elif cfg.ACTIVE_LEARNING.GREEDY_SELECTION:
        al_name = f"greedy_{al_name}"

    if "sampling" in al_name:
        if args.softmax_temp_policy == "constant":
            al_name = f"{al_name}_sm_temp_{args.softmax_temp}"
        else:
            if args.softmax_temp_policy == "exponential":
                al_name = f"{al_name}_sm_temp_policy_{args.softmax_temp_policy}_a_{args.softmax_temp_policy_a}_b_{args.softmax_temp_policy_b}_k_{args.softmax_temp_policy_k}"
            else:
                al_name = f"{al_name}_sm_temp_policy_{args.softmax_temp_policy}_a_{args.softmax_temp_policy_a}_b_{args.softmax_temp_policy_b}"

    if "probcover" in al_name or "dcom" in al_name:
        if cfg.ACTIVE_LEARNING.USE_COSINE_DIST:
            al_name = f"{al_name}_cosine"
        else:
            al_name = f"{al_name}_euclidean"

    # ---- TODO refactor this
    if "probcover" in al_name or "dcom" in al_name:
        initial_delta = args.initial_delta
        if initial_delta is None:
            initial_delta = set_initial_delta(cfg)
        al_name = f"{al_name}_init_delta_{initial_delta}"
    # ----
    if "probcover" in al_name:
        delta_policy = args.delta_policy
        if args.delta_policy != "constant":
            if args.delta_softening_policy is not None:
                delta_policy = f"{delta_policy}_{args.delta_softening_policy}"
            if args.delta_consider_noise:
                delta_policy = f"{delta_policy}_consider_noise"
            al_name = f"{al_name}_{delta_policy}"

    if cfg.ACTIVE_LEARNING.NOISE_DROPOUT:
        al_name = f"{al_name}_dropout"
        # if args.update_features_in_cr:
        #     al_name = f"{al_name}_update_features"
        # elif args.cr_delta != cfg.ACTIVE_LEARNING.INITIAL_DELTA:
        #     al_name = f"{al_name}_delta_{args.cr_delta}"

    if args.change_al_to is not None:
        al_name = f"{al_name}_to_{args.change_al_to}_after_{args.change_al_after}"

    lnl_name = args.lnl
    if cfg.NOISE.NEIGHBORS_FOR_THRESHOLD:
        lnl_name = f"{lnl_name}_neighbors"

    if args.train_on_all_labeled_data:
        train_size_policy = "train_on_all_labeled_data_True"
    elif args.train_on_most_confident:
        train_size_policy = "train_on_most_confident"
    else:
        train_size_policy = "train_on_all_labeled_data_False"

    model = f"{cfg.MODEL.TYPE}_pretrained" if cfg.MODEL.PRETRAINED else cfg.MODEL.TYPE

    if cfg.EXP_NAME == 'auto':
        exp_dir = f'{cfg.DATASET.NAME}_{cfg.DATASET.REPRESENTATION_MODEL}/'\
                  f'{model}/' \
                  f'al_{al_name}/' \
                  f'lnl_{lnl_name}/' \
                  f'noise_mom_coeff_{args.noise_mom_coeff}/' \
                  f'noise_type_{args.noise_type}/' \
                  f'noise_rate_{cfg.NOISE.NOISE_RATE}/' \
                  f'{train_size_policy}/' \
                  f'max_epoch_{args.max_epoch}/' \
                  f'class_balance_{args.balance_classes}/' \
                  f'seed_{args.seed}'
    else:
        exp_dir = cfg.EXP_NAME
    print("Experiment Directory is {}.\n".format(exp_dir))
    logger.info("Experiment Directory is {}.\n".format(exp_dir))

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
    train_data, train_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=True, isDownload=True, representation_model=cfg.DATASET.REPRESENTATION_MODEL, project_features=cfg.MODEL.USE_1NN)
    if "probcover" in al_name and train_size >= 500_000:
    # if train_size >= 500_000:
        print("Creating a subset of the training data for faster computation.")
        subset_size = int(0.1 * train_size)
        indices = np.random.choice(train_size, size=subset_size, replace=False)
        noisy_labels = np.asarray(train_data.noisy_labels)[indices]
        targets = np.asarray(train_data.targets)[indices]
        is_noisy = np.asarray(train_data.is_noisy)[indices]
        noise_rate = cfg.NOISE.NOISE_RATE
        features = train_data.features[indices]
        classes = train_data.classes
        train_data = Subset(train_data, indices)
        train_size = len(train_data)

        # Update the dataset with the new subset
        train_data.targets = targets
        train_data.noisy_labels = noisy_labels
        train_data.is_noisy = is_noisy
        train_data.noise_rate = noise_rate
        train_data.features = features
        train_data.classes = classes
    test_data, test_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=False, isDownload=True, representation_model=cfg.DATASET.REPRESENTATION_MODEL, project_features=cfg.MODEL.USE_1NN)
    cfg.ACTIVE_LEARNING.INIT_L_RATIO = args.initial_size / train_size
    print("\nDataset {} Loaded Sucessfully.\nTotal Train Size: {} and Total Test Size: {}\n".format(cfg.DATASET.NAME, train_size, test_size))
    logger.info("Dataset {} Loaded Sucessfully. Total Train Size: {} and Total Test Size: {}\n".format(cfg.DATASET.NAME, train_size, test_size))

    # Initialize labeled, unlabeled and validation sets
    if cfg.ACTIVE_LEARNING.LSET_PATH is not None:
        print(f"\nLoading Labeled Set from the provided path: {cfg.ACTIVE_LEARNING.LSET_PATH}\n")
        lSet = np.load(cfg.ACTIVE_LEARNING.LSET_PATH, allow_pickle=True)
        uSet = np.setdiff1d(np.arange(train_size), lSet).astype(int)
        cfg.ACTIVE_LEARNING.BUDGET_PER_ROUND[0] -= len(lSet)
    else:
        print("\nCreating Labeled and Unlabeled Sets.\n")  # TODO refactor this - always create empty l_set
        lSet_path, uSet_path, valSet_path = data_obj.makeLUVSets(train_split_ratio=cfg.ACTIVE_LEARNING.INIT_L_RATIO, val_split_ratio=cfg.DATASET.VAL_RATIO, data=train_data, seed_id=cfg.RNG_SEED, save_dir=cfg.EXP_DIR)
        cfg.ACTIVE_LEARNING.LSET_PATH = lSet_path
        cfg.ACTIVE_LEARNING.USET_PATH = uSet_path
        cfg.ACTIVE_LEARNING.VALSET_PATH = valSet_path
        # TODO fix this function so that there will be no race condition
        lSet, uSet, _ = data_obj.loadPartitions(lSetPath=cfg.ACTIVE_LEARNING.LSET_PATH, uSetPath=cfg.ACTIVE_LEARNING.USET_PATH, valSetPath = cfg.ACTIVE_LEARNING.VALSET_PATH)

    # Initialize active learning (AL) and learning with noisy labels (LNL) objects
    lnl_obj = LearningWithNoisyLabels(train_data, test_data, cfg)
    # al_obj = ActiveLearning(data_obj, lnl_obj, cfg)
    test_loader = data_obj.getTestLoader(data=test_data, test_batch_size=cfg.TRAIN.BATCH_SIZE, seed_id=cfg.RNG_SEED)
    l_set_predicted_clean_indices = None
    run_DCoM = cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ['dcom', 'dcom_noisyoracle', 'dcom_noisy_oracle', 'dpc']
    run_ProbCover = "probcover" in cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() or "prob_cover" in cfg.ACTIVE_LEARNING.SAMPLING_FN.lower()

    if run_DCoM and cfg.ACTIVE_LEARNING.INITIAL_DELTA is None:
        cfg.ACTIVE_LEARNING.INITIAL_DELTA = set_initial_delta(cfg, train_data.features)

    delta_scheduler = None
    if run_ProbCover:
        delta_scheduler = DeltaScheduler(cfg, train_data.features)

    # Initialize the model and optimizer
    model = model_builder.build_model(cfg)
    optimizer = optim.construct_optimizer(cfg, model)
    opt_init_state = deepcopy(optimizer.state_dict())
    model_init_state = deepcopy(model.state_dict().copy())
    checkpoint_file = None
    last_model = None  # Just for interpreter

    print(f"Dataset: {cfg.DATASET.NAME}")
    print(f"Representation Model: {cfg.DATASET.REPRESENTATION_MODEL}")
    print(f"AL Query Method: {al_name}")
    print(f"Max AL Episodes: {len(cfg.ACTIVE_LEARNING.BUDGET_PER_ROUND)}")
    print(f"Noise Type: {cfg.NOISE.NOISE_TYPE}")
    if cfg.NOISE.NOISE_TYPE in ["sym", "asym"]:
        noise_rate = cfg.NOISE.NOISE_RATE
    else:
        noise_rate = train_data.noise_rate
    print(f"Noise rate: {noise_rate}")
    print(f"LNL algorithm: {lnl_name}")
    print(f"model: {cfg.MODEL.TYPE}")
    print(f"optimizer: {optimizer}")
    print(f"Class Balance: {cfg.TRAIN.BALANCE_CLASSES}")
    print(f"Max Epoch: {cfg.OPTIM.MAX_EPOCH}")

    logger.info(f"Dataset: {cfg.DATASET.NAME}")
    logger.info(f"Representation Model: {cfg.DATASET.REPRESENTATION_MODEL}")
    logger.info(f"AL Query Method: {al_name}")
    logger.info(f"Max AL Episodes: {len(cfg.ACTIVE_LEARNING.BUDGET_PER_ROUND)}")
    logger.info(f"Noise Type: {cfg.NOISE.NOISE_TYPE}")
    logger.info(f"Noise rate: {noise_rate}")
    logger.info(f"LNL algorithm: {lnl_name}")
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
    coverage_per_episode = []
    delta_avg_lst = []  # only for DCoM
    delta_std_lst = []  # only for DCoM
    delta_per_episode = []  # Only for ProbCover
    softmax_temperatures = []  # Only for ProbCoverSampling
    empty_classes = []
    expected_clean_samples_per_class = ((np.cumsum(cfg.ACTIVE_LEARNING.BUDGET_PER_ROUND) / cfg.MODEL.NUM_CLASSES) * (1 - cfg.NOISE.NOISE_RATE)).astype(int)

    cur_episode = 0
    episode_dir = os.path.join(cfg.EXP_DIR, f'expected_clean_spc_{expected_clean_samples_per_class[cur_episode]}')

    # Try to load the episode if it already exists
    if not args.force_run:
        print("FORCE_RUN is False. Checking if the episode already exists.")
        while os.path.exists(episode_dir):
            print(f"Episode {cur_episode} already exists. Loading the episode and continuing with the next episode.")
            try:
                lSet_temp = np.load(os.path.join(episode_dir, 'lSet.npy'), allow_pickle=True).astype(int)
                uSet_temp = np.load(os.path.join(episode_dir, 'uSet.npy'), allow_pickle=True).astype(int)
                activeSet_temp = np.load(os.path.join(episode_dir, 'activeSet.npy'), allow_pickle=True).astype(int)
                test_accuracies_temp = np.load(os.path.join(episode_dir, 'test_accuracies.npy'), allow_pickle=True).tolist()
                clean_true_positives_temp = np.load(os.path.join(episode_dir, 'clean_true_positives.npy'), allow_pickle=True).tolist()
                clean_false_positives_temp = np.load(os.path.join(episode_dir, 'clean_false_positives.npy'), allow_pickle=True).tolist()
                clean_false_negatives_temp = np.load(os.path.join(episode_dir, 'clean_false_negatives.npy'), allow_pickle=True).tolist()
                clean_true_negatives_temp = np.load(os.path.join(episode_dir, 'clean_true_negatives.npy'), allow_pickle=True).tolist()
                delta_lst_temp = delta_avg_lst_temp = delta_std_lst_temp = None
                if run_DCoM:
                    delta_lst_temp = np.load(os.path.join(episode_dir, 'delta_list.npy'), allow_pickle=True).tolist()
                    delta_avg_lst_temp = np.load(os.path.join(episode_dir, 'avg_delta_per_episode_list.npy'), allow_pickle=True).tolist()
                    delta_std_lst_temp = np.load(os.path.join(episode_dir, 'std_delta_per_episode_list.npy'), allow_pickle=True).tolist()
                elif run_ProbCover:
                    delta_lst_temp = np.load(os.path.join(episode_dir, 'delta_per_episode_list.npy'), allow_pickle=True).tolist()
                    try:
                        softmax_temperatures_temp = np.load(os.path.join(episode_dir, 'softmax_temperatures.npy'), allow_pickle=True).tolist()
                    except FileNotFoundError:
                        softmax_temperatures_temp = [cfg.ACTIVE_LEARNING.SOFTMAX_TEMPERATURE] * len(delta_lst_temp)
                lnl_obj.load_checkpoint(episode_dir)
                if cfg.NOISE.FILTERING_FN == "ideal":
                    l_set_predicted_clean_indices_temp = np.logical_not(train_data.is_noisy[lSet_temp])
                else:
                    l_set_predicted_clean_indices_temp = lnl_obj.get_l_set_is_clean(lSet_temp)

                # OK we're fine with loading the files
                lSet, uSet, activeSet, l_set_predicted_clean_indices = lSet_temp, uSet_temp, activeSet_temp, l_set_predicted_clean_indices_temp
                test_accuracies, clean_true_positives, clean_false_positives, clean_false_negatives, clean_true_negatives = \
                    test_accuracies_temp, clean_true_positives_temp, clean_false_positives_temp, clean_false_negatives_temp, clean_true_negatives_temp
                if run_DCoM:
                    cfg.ACTIVE_LEARNING.DELTA_LST = delta_lst_temp
                    delta_avg_lst = delta_avg_lst_temp
                    delta_std_lst = delta_std_lst_temp
                elif run_ProbCover:
                    delta_per_episode = delta_lst_temp
                    delta_scheduler.current_delta = delta_lst_temp[-1]
                    softmax_temperatures = softmax_temperatures_temp
                    # TODO save and load delta history

                cur_episode += 1
                if cur_episode == len(expected_clean_samples_per_class):
                    break
                episode_dir = os.path.join(cfg.EXP_DIR, f'expected_clean_spc_{expected_clean_samples_per_class[cur_episode]}')

            except Exception as e:
                print(f"Error loading episode {cur_episode}: {e}\nStarting from episode {cur_episode}\n")
                error_traceback = traceback.format_exc()
                # print(error_traceback)
                logger.error(error_traceback)
                break
        if 0 < len(clean_true_positives) < cfg.ACTIVE_LEARNING.MAX_ITER:
            try:
                for i in range(len(clean_true_positives)):
                    accuracy, clean_precision, clean_recall, clean_f1, noise_precision, noise_recall, noise_f1, true_noise, predicted_noise = \
                        calc_metrics(clean_true_positives[i], clean_false_positives[i], clean_true_negatives[i], clean_false_negatives[i])
                    true_noise_ratios.append(true_noise)
                    predicted_noise_ratios.append(predicted_noise)
                    noise_accuracy_per_episode.append(accuracy)
                    clean_recall_per_episode.append(clean_recall)
                    clean_precision_per_episode.append(clean_precision)
                    clean_f1_scores.append(clean_f1)
                    noise_recall_per_episode.append(noise_recall)
                    noise_precision_per_episode.append(noise_precision)
                    noise_f1_scores.append(noise_f1)
            except ZeroDivisionError as e:
                print(f"Error calculating metrics: {e}")
                logger.error(f"Error calculating metrics: {e}")
    else:
        print("FORCE_RUN is True. Starting from scratch.\n")

    # Start Active Learning cycle
    while cur_episode < cfg.ACTIVE_LEARNING.MAX_ITER:
        print("======== EPISODE {} BEGINS ========\n".format(cur_episode))
        logger.info("======== EPISODE {} BEGINS ========\n".format(cur_episode))

        # Creating output directory for the episode
        episode_dir = os.path.join(cfg.EXP_DIR, f'expected_clean_spc_{expected_clean_samples_per_class[cur_episode]}')
        if not os.path.exists(episode_dir):
            os.mkdir(episode_dir)
        cfg.EPISODE_DIR = episode_dir

        print("\n======== ACTIVE SAMPLING ========\n")
        logger.info("\n======== ACTIVE SAMPLING ========\n")
        curr_budget_size = cfg.ACTIVE_LEARNING.BUDGET_PER_ROUND[cur_episode]

        # if expected_clean_samples_per_class[cur_episode] > 10:  # TODO do it a single time
        #     if args.switch_to_random:  # TODO move to cfg
        #         cfg.ACTIVE_LEARNING.SAMPLING_FN = "random"
        #         print("(!) ---------------> Sampling function is set to Random <--------------- (!)")
        #         logger.info("(!) ---------------> Sampling function is set to Random <--------------- (!)")
        #     if args.switch_to_cr:  # TODO move to cfg
        #         cfg.ACTIVE_LEARNING.SAMPLING_FN = "cr"
        #
        #         if args.update_features_in_cr:
        #             print("Extracting features from the model for Conditional Random sampling")
        #             train_features = extract_features_from_model(model, train_data)
        #             # test_features = extract_features_from_model(model, test_data)
        #             set_new_features(True, cfg.DATASET.NAME, cfg.MODEL.TYPE, train_features, cfg.EXP_DIR)
        #             # set_new_features(False, cfg.DATASET.NAME, cfg.MODEL.TYPE, test_features, cfg.EXP_DIR)
        #             cfg.DATASET.REPRESENTATION_MODEL = cfg.MODEL.TYPE
        #             norm_train_features = train_features / np.linalg.norm(train_features, axis=1, keepdims=True)
        #             delta = choose_delta_for_probcover(norm_train_features, cfg.MODEL.NUM_CLASSES)
        #             cfg.ACTIVE_LEARNING.INITIAL_DELTA = delta
        #         else:
        #             print(f"Keep the {cfg.DATASET.REPRESENTATION_MODEL} representations. Setting delta to the provided value")
        #             cfg.ACTIVE_LEARNING.INITIAL_DELTA = args.cr_delta
        #
        #         print(f"(!) ---------------> Sampling function is set to ConditionalRandom. delta is now {cfg.ACTIVE_LEARNING.INITIAL_DELTA} <--------------- (!)")
        #         logger.info(f"(!) ---------------> Sampling function is set to ConditionalRandom. delta is now {cfg.ACTIVE_LEARNING.INITIAL_DELTA} <--------------- (!)")
        #
        if args.change_al_to is not None and expected_clean_samples_per_class[cur_episode] > args.change_al_after:
            if cfg.ACTIVE_LEARNING.SAMPLING_FN != args.change_al_to:
                cfg.ACTIVE_LEARNING.SAMPLING_FN = args.change_al_to
                print(f"(!) ---------------> Sampling function is set to {args.change_al_to} <--------------- (!)")
                logger.info(f"(!) ---------------> Sampling function is set to {args.change_al_to} <--------------- (!)")
                run_DCoM = cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ['dcom', 'dcom_noisyoracle', 'dcom_noisy_oracle', 'dpc']
                run_ProbCover = "probcover" in cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() or "prob_cover" in cfg.ACTIVE_LEARNING.SAMPLING_FN.lower()

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

        # # TODO delete this
        # spc = (len(lSet) / cfg.MODEL.NUM_CLASSES) * (1 - cfg.NOISE.NOISE_RATE)
        # if spc > 30:
        #     delta_scheduler.current_delta = 0.1
        #     print(f"-----------> Update delta to {delta_scheduler.current_delta} <-----------")

        al_obj = ActiveLearning(data_obj, lnl_obj, cfg, delta_scheduler=delta_scheduler)  # TODO check if necessary to recreate the object every episode
        num_shares = 1
        if cfg.ACTIVE_LEARNING.SUPER_GREEDY_SELECTION:
            num_shares = curr_budget_size
        elif cfg.ACTIVE_LEARNING.GREEDY_SELECTION:
            num_shares = np.ceil(curr_budget_size / cfg.MODEL.NUM_CLASSES).astype(int)
        activeSet, new_uSet = al_obj.sample_from_uSet(model, lSet, uSet, train_data, budget_size=curr_budget_size, lSetCleanIdx=l_set_predicted_clean_indices, num_shares=num_shares, training_model=model)
        lSet = np.append(lSet, activeSet).astype(int)  # Add activeSet to lSet, save new_uSet as uSet and update dataloader for the next episode
        uSet = new_uSet.astype(int)
        if run_ProbCover:
            delta_per_episode.append(delta_scheduler.current_delta)
            softmax_temperatures.append(cfg.ACTIVE_LEARNING.SOFTMAX_TEMPERATURE)

        freq = np.bincount(np.take(train_data.targets, lSet.astype(int)))
        if cfg.MODEL.NUM_CLASSES <= 20:
            print(f"Labeled Set Distribution: {freq}.")
            logger.info(f"Labeled Set Distribution: {freq}.")
        num_empty_classes = np.sum(freq == 0)
        print(f"Num of empty classes: {num_empty_classes}")
        logger.info(f"Num of empty classes: {num_empty_classes}")
        empty_classes.append(num_empty_classes)

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

        # Filter Noisy Samples
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

        # calculate coverage
        # coverage = al_obj.calculate_coverage(lSet, uSet, l_set_true_clean_indices)
        # coverage_per_episode.append(coverage)

        # Train model
        if checkpoint_file is not None and os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        if lnl_model is not None and cfg.MODEL.USE_LNL_MODEL is True:
            print("\n======== TRAINING - Done at the Noise Filtering ========")
            logger.info("\n======== TRAINING - Done at the Noise Filtering ========")
            model = lnl_model
            train_loss = lnl_train_loss
        else:
            print("\n======== TRAINING ========")
            logger.info("\n======== TRAINING ========")

            if args.train_on_all_labeled_data:
                train_set = lSet
            elif args.train_on_most_confident:
                train_set = lSet[np.argsort(scores)][int(cfg.NOISE.NOISE_RATE * len(lSet)):]
            else:
                train_set = lSet[l_set_predicted_clean_indices]

            if cfg.MODEL.USE_1NN:  # TODO move to train_model function
                data = np.take(train_data.features, train_set, axis=0)
                labels = np.take(train_data.noisy_labels, train_set)
                knn = KNeighborsClassifier(n_neighbors=1)
                knn.fit(data, labels)
                train_loss = 0
            else:
                if cfg.TRAIN.BALANCE_CLASSES:
                    lSet_loader = data_obj.getBalancedIndexesDataLoader(indexes=train_set, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
                else:
                    lSet_loader = data_obj.getIndexesDataLoader(indexes=train_set, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)

                train_loss = train_model(lSet_loader, model, optimizer, cfg)

                # print("Extract features from the model and save them in the episode directory")
                # features = extract_features_from_model(model, train_data)
                # np.save(os.path.join("/cs/labs/daphna/nettashaf/TypiClustNoisy/experiments/19_conditional_random_sampling_sym_noise", f'features_spc_{args.samples_per_class[cur_episode]}.npy'), features)

        # Save the model
        model.eval()
        last_model = model
        last_model_state = model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict()
        checkpoint_file = cu.save_checkpoint(info="train_loss_" + str(round(train_loss, 4)), model_state=last_model_state, optimizer_state=optimizer.state_dict(), epoch=cfg.OPTIM.MAX_EPOCH, cfg=cfg)
        print('\nWrote Last Model Checkpoint to: {}\n'.format(checkpoint_file.split('/')[-1]))
        logger.info('\nWrote Last Model Checkpoint to: {}\n'.format(checkpoint_file))

        print("Extracting features from the model")
        train_features = extract_features_from_model(model, train_data)
        # test_features = extract_features_from_model(model, test_data)
        set_new_features(True, cfg.DATASET.NAME, cfg.DATASET.REPRESENTATION_MODEL, train_features, cfg.EXP_DIR)
        # set_new_features(False, cfg.DATASET.NAME, cfg.MODEL.TYPE, test_features, cfg.EXP_DIR)
        # cfg.DATASET.REPRESENTATION_MODEL = cfg.MODEL.TYPE
        # norm_train_features = train_features / np.linalg.norm(train_features, axis=1, keepdims=True)
        # delta = choose_delta_for_probcover(norm_train_features, cfg.MODEL.NUM_CLASSES, True)

        # Test last model checkpoint
        print("\n======== TESTING ========\n")
        logger.info("\n======== TESTING ========\n")
        if cfg.MODEL.USE_1NN:  # TODO move to test_model function
            predictions = knn.predict(test_data.features)
            test_acc = 100. * (predictions == np.array(test_data.targets)).mean()
        else:
            test_acc = test_model(test_loader, cfg, cur_episode, model, checkpoint_file)
        test_accuracies.append(test_acc)

        # print episode stats
        print("EPISODE {} Test Accuracy: {}.\n".format(cur_episode, round(test_acc, 4)))
        logger.info("EPISODE {} Test Accuracy {}.\n".format(cur_episode, test_acc))

        headers = [f"{spc} SPC" for spc in expected_clean_samples_per_class[:cur_episode + 1]]
        data_for_table = [np.asarray(test_accuracies).round(3), np.cumsum(cfg.ACTIVE_LEARNING.BUDGET_PER_ROUND[:cur_episode+1])]
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
        if len(empty_classes) == len(test_accuracies):
            data_for_table.append(np.asarray(empty_classes))
            rows.append("Empty Classes")
        if run_DCoM:
            data_for_table.append(np.asarray(delta_avg_lst).round(4))
            data_for_table.append(np.asarray(delta_std_lst).round(4))
            rows.append("Delta Avg")
            rows.append("Delta Std")
        if run_ProbCover:
            data_for_table.append(np.asarray(delta_per_episode).round(5))
            rows.append("ProbCover Delta")
        if "sampling" in cfg.ACTIVE_LEARNING.SAMPLING_FN.lower():
            data_for_table.append(np.asarray(softmax_temperatures).round(3))
            rows.append("Softmax Temperature")
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
            np.save(os.path.join(cfg.EPISODE_DIR, 'softmax_temperatures.npy'), softmax_temperatures)

        cur_episode += 1

        print("\n================================\n\n")
        logger.info("\n================================\n\n")

        # if not cfg.ACTIVE_LEARNING.FINE_TUNE:
        # start model from scratch
        print('Starting model from scratch - ignoring existing weights.')
        logger.info('Starting model from scratch - ignoring existing weights.')
        model = model_builder.build_model(cfg)
        optimizer = optim.construct_optimizer(cfg, model)
        print(model.load_state_dict(model_init_state))
        print(optimizer.load_state_dict(opt_init_state))

    save_stats(cfg.EXP_DIR, test_accuracies, clean_true_positives, clean_false_positives, clean_false_negatives, clean_true_negatives)
    print("Saved the final accuracies and metrics")
    logger.info("Saved the final accuracies and metrics")


def train_model(train_loader, model, optimizer, cfg):
    global plot_episode_xvalues
    global plot_episode_yvalues

    global plot_epoch_xvalues
    global plot_epoch_yvalues

    global plot_it_x_values
    global plot_it_y_values

    start_epoch = 0
    loss_fun = losses.get_loss_fun()

    # Create meters
    train_meter = TrainMeter(len(train_loader))

    # Perform the training loop
    # print("Len(train_loader):{}".format(len(train_loader)))

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

        plot_epoch_xvalues.append(cur_epoch+1)
        plot_epoch_yvalues.append(train_loss)

        # save_plot_values([plot_epoch_xvalues, plot_epoch_yvalues, plot_it_x_values, plot_it_y_values, val_acc_epochs_x, val_acc_epochs_y],\
        #     ["plot_epoch_xvalues", "plot_epoch_yvalues", "plot_it_x_values", "plot_it_y_values","val_acc_epochs_x","val_acc_epochs_y"], out_dir=cfg.EPISODE_DIR, isDebug=False)
        # logger.info("Successfully logged numpy arrays!!")

        # Plot arrays
        # plot_arrays(x_vals=plot_epoch_xvalues, y_vals=plot_epoch_yvalues, \
        # x_name="Epochs", y_name="Loss", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR)
        #
        # plot_arrays(x_vals=val_acc_epochs_x, y_vals=val_acc_epochs_y, \
        # x_name="Epochs", y_name="Validation Accuracy", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR)

        # save_plot_values([plot_epoch_xvalues, plot_epoch_yvalues, plot_it_x_values, plot_it_y_values, val_acc_epochs_x, val_acc_epochs_y], \
        #         ["plot_epoch_xvalues", "plot_epoch_yvalues", "plot_it_x_values", "plot_it_y_values","val_acc_epochs_x","val_acc_epochs_y"], out_dir=cfg.EPISODE_DIR)
        if cur_epoch == 0 or (cur_epoch + 1) % 25 == 0:
            print('Training Epoch: {}/{}\tTrain Loss: {}\t'.format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH, round(train_loss, 4)))
            logger.info('Training Epoch: {}/{}\tTrain Loss: {}\t'.format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH, round(train_loss, 4)))

    # plot_arrays(x_vals=plot_epoch_xvalues, y_vals=plot_epoch_yvalues, \
    #     x_name="Epochs", y_name="Loss", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR)
    #
    # plot_arrays(x_vals=plot_it_x_values, y_vals=plot_it_y_values, \
    #     x_name="Iterations", y_name="Loss", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR)
    #
    # plot_arrays(x_vals=val_acc_epochs_x, y_vals=val_acc_epochs_y, \
    #     x_name="Epochs", y_name="Validation Accuracy", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR)

    plot_epoch_xvalues = []
    plot_epoch_yvalues = []
    plot_it_x_values = []
    plot_it_y_values = []

    return train_loss


def test_model(test_loader, cfg, cur_episode, model, checkpoint_file=None):

    global plot_episode_xvalues
    global plot_episode_yvalues

    global plot_epoch_xvalues
    global plot_epoch_yvalues

    global plot_it_x_values
    global plot_it_y_values

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

    # --- alternative way to calculate test accuracy
    # model.cuda()
    # model.eval()
    # predictions = model(torch.Tensor(test_loader.dataset.features).cuda()).argmax(axis=1).cpu().numpy()
    # test_acc = 100. * (predictions == np.array(test_loader.dataset.targets)).mean()

    plot_episode_xvalues.append(cur_episode)
    plot_episode_yvalues.append(test_acc)

    # plot_arrays(x_vals=plot_episode_xvalues, y_vals=plot_episode_yvalues, \
    #     x_name="Episodes", y_name="Test Accuracy", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EXP_DIR)
    #
    # save_plot_values([plot_episode_xvalues, plot_episode_yvalues], \
    #     ["plot_episode_xvalues", "plot_episode_yvalues"], out_dir=cfg.EXP_DIR)

    return test_acc


def train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch, cfg, clf_iter_count, clf_change_lr_iter, clf_max_iter):
    """Performs one epoch of training."""
    global plot_episode_xvalues
    global plot_episode_yvalues

    global plot_epoch_xvalues
    global plot_epoch_yvalues

    global plot_it_x_values
    global plot_it_y_values

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
    # train_meter.iter_tic()  # This basically notes the start time in timer class defined in utils/timer.py

    len_train_loader = len(train_loader)
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
        # Combine the stats across the GPUs
        # if cfg.NUM_GPUS > 1:
        #     #Average error and losses across GPUs
        #     #Also this this calls wait method on reductions so we are ensured
        #     #to obtain synchronized results
        #     loss, top1_err = du.scaled_all_reduce(
        #         [loss, top1_err]
        #     )
        # Copy the stats from GPU to CPU (sync point)
        loss, top1_err = loss.item(), top1_err.item()
        # #Only master process writes the logs which are used for plotting
        # if du.is_master_proc():
        if cur_iter != 0 and cur_iter%19 == 0:
            #because cur_epoch starts with 0
            plot_it_x_values.append((cur_epoch)*len_train_loader + cur_iter)
            plot_it_y_values.append(loss)
            # save_plot_values([plot_it_x_values, plot_it_y_values],["plot_it_x_values", "plot_it_y_values"], out_dir=cfg.EPISODE_DIR, isDebug=False)
            # print(plot_it_x_values)
            # print(plot_it_y_values)
            #Plot loss graphs
            # plot_arrays(x_vals=plot_it_x_values, y_vals=plot_it_y_values, x_name="Iterations", y_name="Loss", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR,)
            # print('Training Epoch: {}/{}\tIter: {}/{}'.format(cur_epoch+1, cfg.OPTIM.MAX_EPOCH, cur_iter, len(train_loader)))

        #Compute the difference in time now from start time initialized just before this for loop.
        # train_meter.iter_toc()
        # train_meter.update_stats(top1_err=top1_err, loss=loss, \
        #     lr=lr, mb_size=inputs.size(0) * cfg.NUM_GPUS)
        # train_meter.log_iter_stats(cur_epoch, cur_iter)
        # train_meter.iter_tic()
    # Log epoch stats
    # train_meter.log_epoch_stats(cur_epoch)
    # train_meter.reset()
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

    global plot_episode_xvalues
    global plot_episode_yvalues

    global plot_epoch_xvalues
    global plot_epoch_yvalues

    global plot_it_x_values
    global plot_it_y_values

    if torch.cuda.is_available():
        model.cuda()

    # Enable eval mode
    model.eval()
    # test_meter.iter_tic()

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
            # Combine the errors across the GPUs
            # if cfg.NUM_GPUS > 1:
            #     top1_err = du.scaled_all_reduce([top1_err])
            #     #as above returns a list
            #     top1_err = top1_err[0]
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
            # test_meter.log_iter_stats(cur_epoch, cur_iter)
            # test_meter.iter_tic()
    # Log epoch stats
    # test_meter.log_epoch_stats(cur_epoch)
    # test_meter.reset()

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
    if cfg.DATASET.REPRESENTATION_MODEL != args.rep and args.rep not in [None, ""]:
        cfg.DATASET.REPRESENTATION_MODEL = args.rep
    if cfg.DATASET.REPRESENTATION_DIM is None:
        cfg.DATASET.REPRESENTATION_DIM = 384 if cfg.DATASET.NAME.upper() in ['IMAGENET50', 'IMAGENET100', 'IMAGENET200', 'CLOTHING1M', "FOOD101N"] else 512

    assert cfg.DATASET.VAL_RATIO == 0, "Validation set is not supported in active learning"

    # AL
    cfg.ACTIVE_LEARNING.SAMPLING_FN = args.al
    cfg.ACTIVE_LEARNING.USE_COSINE_DIST = args.cosine_dist
    cfg.ACTIVE_LEARNING.A_LOGISTIC = args.a_logistic
    cfg.ACTIVE_LEARNING.K_LOGISTIC = args.k_logistic
    cfg.ACTIVE_LEARNING.SUPER_GREEDY_SELECTION = args.super_greedy_selection or args.noisy_label_inference or (args.lnl == "ideal" and "noisy" in args.al.lower())
    cfg.ACTIVE_LEARNING.GREEDY_SELECTION = cfg.ACTIVE_LEARNING.SUPER_GREEDY_SELECTION or args.greedy_selection or "noisy" in args.al.lower()
    cfg.ACTIVE_LEARNING.LNL_TRAIN_ON_GREEDY_SELECTION = not args.noisy_label_inference
    cfg.ACTIVE_LEARNING.DELTA_LST = []  # relevant for DCoM only
    cfg.ACTIVE_LEARNING.NOISE_DROPOUT = args.use_noise_dropout

    cfg.ACTIVE_LEARNING.SOFTMAX_TEMPERATURE = args.softmax_temp
    cfg.ACTIVE_LEARNING.TEMPERATURE_POLICY.MAJOR_POLICY = args.softmax_temp_policy
    cfg.ACTIVE_LEARNING.TEMPERATURE_POLICY.A = args.softmax_temp_policy_a
    cfg.ACTIVE_LEARNING.TEMPERATURE_POLICY.B = args.softmax_temp_policy_b
    cfg.ACTIVE_LEARNING.TEMPERATURE_POLICY.K = args.softmax_temp_policy_k

    if args.initial_delta is not None:
        cfg.ACTIVE_LEARNING.INITIAL_DELTA = args.initial_delta
    else:
        cfg.ACTIVE_LEARNING.INITIAL_DELTA = None
    cfg.ACTIVE_LEARNING.DELTA_POLICY.MAJOR_POLICY = args.delta_policy
    cfg.ACTIVE_LEARNING.DELTA_POLICY.SOFTENING_POLICY = args.delta_softening_policy
    cfg.ACTIVE_LEARNING.DELTA_POLICY.CONSIDER_NOISE = args.delta_consider_noise
    cfg.ACTIVE_LEARNING.MAX_DELTA = 0.5 if args.cosine_dist else 1.1
    # cfg.ACTIVE_LEARNING.DELTA_RESOLUTION = 0.05 if args.project_features else 0.02
    cfg.ACTIVE_LEARNING.DELTA_RESOLUTION = 0.02

    # LNL
    cfg.NOISE.FILTERING_FN = args.lnl
    if cfg.DATASET.NAME in ["CLOTHING1M"]:
        cfg.NOISE.NOISE_TYPE = "noisy_label"
    else:
        cfg.NOISE.NOISE_TYPE = args.noise_type
    if cfg.NOISE.NOISE_TYPE in ['sym', 'asym']:
        cfg.NOISE.NOISE_RATE = args.noise_rate
    else:
        noise_dict = {"CIFAR100N": {"noisy_label": 0.402, "clean_label": 0},
                      "CIFAR10N": {'clean_label': 0, 'worse_label': 0.4021, 'aggre_label': 0.0901,
                                   'random_label1': 0.1723, 'random_label2': 0.1812, 'random_label3': 0.1764},
                      "CLOTHING1M": {"noisy_label": 0.38},
                      "FOOD101N": {"noisy_label": 0.2},
                      }
        cfg.NOISE.NOISE_RATE = noise_dict[cfg.DATASET.NAME][cfg.NOISE.NOISE_TYPE]
    cfg.NOISE.MOMENTUM_COEFFICIENT = args.noise_mom_coeff
    cfg.NOISE.NEIGHBORS_FOR_THRESHOLD = args.use_neighbors_for_threshold

    # Model & Training
    cfg.MODEL.USE_LNL_MODEL = args.use_lnl_model
    cfg.MODEL.USE_1NN = args.use_1nn
    if cfg.MODEL.USE_1NN:
        cfg.MODEL.TYPE = '1NN'
    cfg.MODEL.LINEAR_FROM_FEATURES = args.linear_from_features
    if cfg.MODEL.LINEAR_FROM_FEATURES:
        cfg.MODEL.TYPE = 'linear'
    cfg.OPTIM.MAX_EPOCH = args.max_epoch
    cfg.TRAIN.BALANCE_CLASSES = cfg.TRAIN.BALANCE_CLASSES or args.balance_classes

    # Budget
    cfg.ACTIVE_LEARNING.LSET_PATH = args.l_set_path
    if args.budget > 0:
        cfg.ACTIVE_LEARNING.INITIAL_BUDGET_SIZE = args.initial_budget if args.initial_budget > 0 else args.budget
        cfg.ACTIVE_LEARNING.BUDGET_SIZE = args.budget
        budget_per_round = [args.budget] * args.num_episodes
        budget_per_round[0] = cfg.ACTIVE_LEARNING.INITIAL_BUDGET_SIZE
    elif len(args.samples_per_class) > 0:
        budgets = [get_budget(cfg.NOISE.NOISE_RATE, spc, cfg.MODEL.NUM_CLASSES) for spc in args.samples_per_class]
        budget_per_round = [budgets[0]] + [budgets[i] - budgets[i-1] for i in range(1, len(budgets))]
    elif len(args.cumulative_budget) > 0:
        budget_per_round = [args.cumulative_budget[0]] + [args.cumulative_budget[i] - args.cumulative_budget[i-1] for i in range(1, len(args.cumulative_budget))]
    else:
        raise ValueError('Either budget or samples_per_class should be provided')
    cfg.ACTIVE_LEARNING.MAX_ITER = len(budget_per_round)
    cfg.ACTIVE_LEARNING.BUDGET_PER_ROUND = budget_per_round

    cfg.DATA_LOADER.NUM_WORKERS = 0  # TODO delete this line

    if args.debug:
        args.force_run = True
        cfg.DATA_LOADER.NUM_WORKERS = 0
        cfg.EXP_NAME = 'test'

    if args.lnl.lower() in ['unicon']:
        cfg.DATA_LOADER.NUM_WORKERS = 0,

    try:
        # for i in range(5):
        main(cfg)
    except Exception as e:
        logger.error(f'Error ({type(e).__name__}): {e}')
        error_traceback = traceback.format_exc()
        logger.error(error_traceback)
        raise e
