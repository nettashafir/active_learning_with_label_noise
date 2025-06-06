import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pycls.utils.logging as lu

logger = lu.get_logger(__name__)

DATASET_FEATURES_DICT = {
    'train':
        {
            'CIFAR10': {"simclr": '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/cifar-10_simclr/pretext/features_seed1.npy'},
            'CIFAR10N': {"simclr": '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/cifar-10_simclr/pretext/features_seed1.npy'},
            'CIFAR100N': {
                "simclr": '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/cifar-100_simclr/cifar100_simclr_train.npy',
                "mocov2plus": '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/cifar-100_mocov2plus/train.npy',
                "barlow_twins": '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/cifar-100_barlow_twins/train.npy',
                "byol": '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/cifar-100_BYOL/train_features.npy',
                "dino": "/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/cifar-100_dino/train_features.pth",
            },

            'CIFAR100': {"simclr": '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/cifar-100_simclr/cifar100_simclr_train.npy',
                         "mocov2plus": '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/cifar-100_mocov2plus/train.npy',
                         "barlow_twins": '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/cifar-100_barlow_twins/train.npy',
                         "byol": '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/cifar-100_BYOL/train_features.npy',
                        "dino": "/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/cifar-100_dino/train_features.pth",
                         },
            # 'CIFAR10': '../../representations/cifar10_simclr_features_seed1.npy',
            # 'CIFAR10N': '../../representations/cifar10_simclr_features_seed1.npy',
            # 'CIFAR100': '../../representations/cifar100_simclr_features_seed1.npy',
            # 'CIFAR100N': '../../representations/cifar100_simclr_features_seed1.npy',
            'TINYIMAGENET': {"simclr": '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/tiny-imagenet_simclr/pretext/features_seed1.npy'},
            'IMAGENET': {'dinov2': '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/imagenet_dinov2/train_features.npy'},
            'IMAGENET50': {'dinov2': '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/imagenet50_dinov2/features_IMAGENET50_dinov2_train.npy',
                           'dinov2_small': '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/imagenet50_dinov2_vits14_reg_pretrained/imagenet50_train_features.pth',
                           'dinov2_small_hf': '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/imagenet50_dinov2-small-imagenet1k-1-layer_pretrained/imagenet50_train_features.pth'},
            # 'IMAGENET100': '../../dino/runs/trainfeat.pth',
            # 'IMAGENET200': '../../dino/runs/trainfeat.pth',
            'CLOTHING1M': {'dinov2_small': '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/clothing1m_dinov2_vits14_reg_pretrained/clothing1m_train_features.pth',
                           'dinov2_base': '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/clothing1m_dinov2_vitb14_reg_pretrained/clothing1m_train_features.pth'},
            'FOOD101N': {'dinov2_small': '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/food101n_dinov2_vits14_reg_pretrained/food101n_train_features.pth'},
            'WEBVISION': {'dinov2_small': '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/clothing1m_dinov2_vits14_reg_pretrained/clothing1m_train_features.pth'},
            'MINIWEBVISION_FLICKR': {'dinov2_small': '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/webvision_dinov2_vits14_reg_pretrained/mini_webvision_flickr_train_features.pth'},
            'MINIWEBVISION_GOOGLE': {'dinov2_small': '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/webvision_dinov2_vits14_reg_pretrained/mini_webvision_google_train_features.pth'}
        },
    'test':
        {
            'CIFAR10': {"simclr": '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/cifar-10_simclr/pretext/test_features_seed1.npy'},
            'CIFAR10N': {"simclr": '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/cifar-10_simclr/pretext/test_features_seed1.npy'},
            'CIFAR100N': {"simclr": '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/cifar-100_simclr/cifar100_simclr_test.npy',
                         "mocov2plus": '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/cifar-100_mocov2plus/test.npy',
                         "barlow_twins": '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/cifar-100_barlow_twins/test.npy',
                         "byol": '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/cifar-100_BYOL/test_features.npy',
                         "dino": "/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/cifar-100_dino/test_features.pth",
                          },
            'CIFAR100': {"simclr": '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/cifar-100_simclr/cifar100_simclr_test.npy',
                         "mocov2plus": '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/cifar-100_mocov2plus/test.npy',
                         "barlow_twins": '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/cifar-100_barlow_twins/test.npy',
                         "byol": '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/cifar-100_BYOL/test_features.npy',
                         "dino": "/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/cifar-100_dino/test_features.pth",
                         },
            'TINYIMAGENET': {"simclr": '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/tiny-imagenet_simclr/pretext/test_features_seed1.npy'},
            'IMAGENET': {'dinov2': '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/imagenet_dinov2/val_features.npy'},
            'IMAGENET50': {'dinov2': '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/imagenet50_dinov2/features_IMAGENET50_dinov2_test.npy',
                           'dinov2_small': '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/imagenet50_dinov2_vits14_reg_pretrained/imagenet50_test_features.pth',
                           'dinov2_small_hf': '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/imagenet50_dinov2-small-imagenet1k-1-layer_pretrained/imagenet50_test_features.pth'},
            # 'IMAGENET100': '../../dino/runs/testfeat.pth',pwd
            # 'IMAGENET200': '../../dino/runs/testfeat.pth',
            'CLOTHING1M': {'dinov2_small': '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/clothing1m_dinov2_vits14_reg_pretrained/clothing1m_test_features.pth',
                           'dinov2_base': '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/clothing1m_dinov2_vitb14_reg_pretrained/clothing1m_test_features.pth'},
            'FOOD101N': {'dinov2_small': '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/food101n_dinov2_vits14_reg_pretrained/food101n_train_features.pth'},
            'WEBVISION': {'dinov2_small': '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/clothing1m_dinov2_vits14_reg_pretrained/clothing1m_test_features.pth'},
            'MINIWEBVISION_FLICKR': {'dinov2_small': '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/webvision_dinov2_vits14_reg_pretrained/webvision_test_features.pth'},
            'MINIWEBVISION_GOOGLE': {'dinov2_small': '/cs/labs/daphna/nettashaf/TypiClustNoisy/representations_bank/webvision_dinov2_vits14_reg_pretrained/webvision_test_features.pth'}
        }
}


def load_features(ds_name, representation_model, seed=1, train=True,
                  normalize=False, project=False, center=False
                  ):
    " load pretrained features for a dataset "
    split = "train" if train else "test"
    fname = DATASET_FEATURES_DICT[split][ds_name][representation_model].format(seed=seed)
    if fname.endswith('.npy'):
        features = np.load(fname)
    elif fname.endswith('.pth'):
        features = torch.load(fname)
    else:
        raise Exception("Unsupported filetype")

    # Features are normalized to have maximum unit norm
    if center:
        print("Features | Centering features")
        logger.info("Centering features")
        features = features - np.mean(features, axis=0)
    if normalize:
        print("Features | Normalizing features to have maximum unit norm")
        logger.info("Normalizing features to have maximum unit norm")
        norms = np.linalg.norm(features, axis=1)
        max_norm = np.max(norms)
        # min_norm = np.min(norms)
        features = features / max_norm
    if project:
        print("Features | Projecting features to the unit sphere")
        logger.info("Projecting features to the unit sphere")
        features = features / np.linalg.norm(features, axis=1, keepdims=True)

    return features


def extract_features_from_model(model: nn.Module, dataset: Dataset):
    # Put model in evaluation mode
    model.eval()

    # Initialize DataLoader
    data_loader = DataLoader(dataset, batch_size=100, shuffle=False)

    # Store features
    features_list = []

    # Register hook to capture the output from the penultimate layer
    def hook(module, input, output):
        features_list.append(output.detach().cpu().reshape(output.shape[0], -1))

    # Register the hook on the layer before the last (usually avgpool for ResNet)
    handle = model.avgpool.register_forward_hook(hook)

    # Pass all images through the model to collect features
    with torch.no_grad():
        for batch in tqdm(data_loader):
            inputs = batch[0]
            inputs = inputs.to("cuda")
            model(inputs)

    # Remove the hook
    handle.remove()

    # Concatenate all features along the batch dimension
    features = torch.cat(features_list, dim=0)
    return features


def set_new_features(is_train, dataset_name, model_name, features, outdir):
    split = "train" if is_train else "test"
    if isinstance(features, np.ndarray):
        filename = os.path.join(outdir, f"features_{dataset_name}_{model_name}_{split}.npy")
        np.save(filename, features)
    elif isinstance(features, torch.Tensor):
        filename = os.path.join(outdir, f"features_{dataset_name}_{model_name}_{split}.pth")
        torch.save(features, filename)
    else:
        raise Exception("Unsupported filetype")
    if dataset_name not in DATASET_FEATURES_DICT[split]:
        DATASET_FEATURES_DICT[split][dataset_name] = {}
    DATASET_FEATURES_DICT[split][dataset_name][model_name] = filename
