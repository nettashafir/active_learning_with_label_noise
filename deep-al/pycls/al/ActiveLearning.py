# This file is slightly modified from a code implementation by Prateek Munjal et al., authors of the paper https://arxiv.org/abs/2002.09564
# GitHub: https://github.com/PrateekMunjal
# ----------------------------------------------------------
import numpy as np
import pandas as pd
import torch

from .Sampling import Sampling, CoreSetMIPSampling, AdversarySampler
import pycls.utils.logging as lu

logger = lu.get_logger(__name__)


class ActiveLearning:
    """
    Implements standard active learning methods.
    """

    def __init__(self, dataObj, lnl_obj, cfg, delta_scheduler=None):
        self.dataObj = dataObj
        self.sampler = Sampling(dataObj=dataObj, cfg=cfg)
        self.lnl_obj = lnl_obj
        self.delta_scheduler = delta_scheduler
        self.cfg = cfg
        self.lnl_train = cfg.ACTIVE_LEARNING.LNL_TRAIN_ON_GREEDY_SELECTION

    def sample_from_uSet(self, clf_model, lSet, uSet, trainDataset, supportingModels=None, lSetCleanIdx=None, budget_size=None):
        """
        Sample from uSet using cfg.ACTIVE_LEARNING.SAMPLING_FN.

        INPUT
        ------
        clf_model: Reference of task classifier model class [Typically VGG]

        supportingModels: List of models which are used for sampling process.

        OUTPUT
        -------
        Returns activeSet, uSet
        """
        # if self.delta_scheduler is not None and len(lSet) > 0:
        #     self.delta_scheduler.update_delta(l_set=lSet, l_set_predicted_is_clean=lSetCleanIdx)
        greedy = self.cfg.ACTIVE_LEARNING.GREEDY_SELECTION
        budget_size = budget_size if budget_size is not None else self.cfg.ACTIVE_LEARNING.BUDGET_SIZE
        assert budget_size > 0, "Expected a positive budgetSize"
        assert budget_size <= len(uSet), "BudgetSet cannot exceed length of unlabelled set. " \
                                        "Length of unlabelled set: {} and budgetSize: {}".format(len(uSet), budget_size)

        if len(lSet) == 0 and self.cfg.ACTIVE_LEARNING.SAMPLING_FN in ["direct", "coreset", "coreset_noisy_oracle", "margin", "badge", "entropy", "uncertainty", "dbal", "bald"]:
            print("AL Object | No labelled data available. Sampling randomly.")
            logger.info("AL Object | No labelled data available. Sampling randomly.")
            activeSet, uSet = self.sampler.random(uSet=uSet, budgetSize=budget_size)
            return activeSet, uSet

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN in ["random", "random2"]:
            print(f"AL Object | Random Sampling of {budget_size} samples from uSet")
            logger.info(f"AL Object | Random Sampling of {budget_size} samples from uSet")
            activeSet, uSet = self.sampler.random(uSet=uSet, budgetSize=budget_size)
        
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "uncertainty":
            oldmode = clf_model.training
            clf_model.eval()
            activeSet, uSet = self.sampler.uncertainty(budgetSize=budget_size,lSet=lSet,uSet=uSet ,model=clf_model,dataset=trainDataset)
            clf_model.train(oldmode)
        
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "entropy":
            oldmode = clf_model.training
            clf_model.eval()
            activeSet, uSet = self.sampler.entropy(budgetSize=budget_size,lSet=lSet,uSet=uSet ,model=clf_model,dataset=trainDataset)
            clf_model.train(oldmode)
        
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "margin":
            oldmode = clf_model.training
            clf_model.eval()
            activeSet, uSet = self.sampler.margin(budgetSize=budget_size,lSet=lSet,uSet=uSet ,model=clf_model,dataset=trainDataset)
            clf_model.train(oldmode)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "coreset":
            waslatent = clf_model.penultimate_active
            wastrain = clf_model.training
            clf_model.penultimate_active = True
            clf_model.eval()
            coreSetSampler = CoreSetMIPSampling(cfg=self.cfg, dataObj=self.dataObj)
            activeSet, uSet = coreSetSampler.query(lSet=lSet, uSet=uSet, clf_model=clf_model, dataset=trainDataset)
            clf_model.penultimate_active = waslatent
            clf_model.train(wastrain)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() == "badge":
            oldmode = clf_model.training
            clf_model.eval()
            waslatent = clf_model.penultimate_active
            clf_model.penultimate_active = True
            activeSet, uSet = self.sampler.badge(budgetSize=budget_size, uSet=uSet, model=clf_model, dataset=trainDataset)
            clf_model.train(oldmode)
            clf_model.penultimate_active = waslatent

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "typiclust":
            from .typiclust import TypiClust
            is_scan = self.cfg.ACTIVE_LEARNING.SAMPLING_FN.endswith('dc')
            tpc = TypiClust(self.cfg, lSet, uSet, budgetSize=budget_size, is_scan=is_scan)
            activeSet, uSet = tpc.select_samples()

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["prob_cover", 'probcover']:
            from .prob_cover import ProbCover
            probcov = ProbCover(self.cfg, lSet, uSet, budgetSize=budget_size, delta_scheduler=self.delta_scheduler)
            activeSet, uSet = probcov.select_samples()

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["dcom"]:
            from .DCoM import DCoM
            dcom = DCoM(self.cfg, lSet, uSet, budgetSize=budget_size,
                        lSet_deltas=self.cfg.ACTIVE_LEARNING.DELTA_LST,
                        max_delta=self.cfg.ACTIVE_LEARNING.MAX_DELTA)
            activeSet, uSet = dcom.select_samples(clf_model, trainDataset, self.dataObj)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["maxherding", "max_herding"]:
            from .maxherding import MaxHerding
            delta = self.cfg.ACTIVE_LEARNING.INITIAL_DELTA
            herding = MaxHerding(self.cfg, lSet, uSet, budget_size, delta=delta)
            activeSet, uSet = herding.select_samples()

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["npc", "probcover_nas"]:
            from .npc import NPC
            assert not (len(lSet) > 0 and lSetCleanIdx is None), "Expected lSetCleanIdx to be provided for noisy oracle"
            if len(lSet) == 0:
                l_set_clean, l_set_noisy = [], []
            else:
                lSetCleanIdx = np.array(lSetCleanIdx)
                l_set_clean = lSet[lSetCleanIdx == 1]
                l_set_noisy = lSet[lSetCleanIdx == 0]
            npc_obj = NPC(self.cfg, l_set_clean, l_set_noisy, uSet, budget_size=budget_size, delta_scheduler=self.delta_scheduler, lnl_obj=self.lnl_obj)
            activeSet, uSet = npc_obj.select_samples()

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["weighted_npc"]:
            from .npc_weighted import GreedyProbCoverNoisyOracleWighted
            assert not (len(lSet) > 0 and lSetCleanIdx is None), "Expected lSetCleanIdx to be provided for noisy oracle"
            if len(lSet) == 0:
                l_set_clean, l_set_noisy = [], []
            else:
                lSetCleanIdx = np.array(lSetCleanIdx)
                l_set_clean = lSet[lSetCleanIdx == 1]
                l_set_noisy = lSet[lSetCleanIdx == 0]
            wnpc_obj = GreedyProbCoverNoisyOracleWighted(self.cfg, l_set_clean, l_set_noisy, uSet, budget_size=budget_size, delta_scheduler=self.delta_scheduler, lnl_obj=self.lnl_obj)
            activeSet, uSet = wnpc_obj.select_samples()

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["maxherding_nas", "max_herding_nas"]:
            from .maxherding_nas import MaxHerdingNas
            delta = self.cfg.ACTIVE_LEARNING.INITIAL_DELTA
            herding_nas = MaxHerdingNas(self.cfg, lSet, uSet, budget_size, lnl_obj=self.lnl_obj, delta=delta)
            activeSet, uSet = herding_nas.select_samples()

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "coreset_nas":
            from .coreset_nas import CoreSetNas
            waslatent = clf_model.penultimate_active
            wastrain = clf_model.training
            clf_model.penultimate_active = True
            clf_model.eval()
            coreSetSampler = CoreSetNas(cfg=self.cfg, dataObj=self.dataObj, lnl_obj=self.lnl_obj)
            activeSet, uSet = coreSetSampler.query(lSet=lSet, uSet=uSet, clf_model=clf_model, dataset=trainDataset)
            clf_model.penultimate_active = waslatent
            clf_model.train(wastrain)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() == "direct":
            from .direct import DIRECT
            direct = DIRECT(self.cfg, lSet, uSet, budget_size, self.dataObj)
            activeSet, uSet = direct.select_samples(clf_model, trainDataset)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "dbal" or self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "DBAL":
            activeSet, uSet = self.sampler.dbal(budgetSize=budget_size,
                uSet=uSet, clf_model=clf_model, dataset=trainDataset)
            
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "bald" or self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "BALD":
            activeSet, uSet = self.sampler.bald(budgetSize=budget_size, uSet=uSet, clf_model=clf_model, dataset=trainDataset)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "ensemble_var_R":
            activeSet, uSet = self.sampler.ensemble_var_R(budgetSize=budget_size, uSet=uSet, clf_models=supportingModels, dataset=trainDataset)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "vaal":
            adv_sampler = AdversarySampler(cfg=self.cfg, dataObj=self.dataObj)

            # Train VAE and discriminator first
            vae, disc, uSet_loader = adv_sampler.vaal_perform_training(lSet=lSet, uSet=uSet, dataset=trainDataset)

            # Do active sampling
            activeSet, uSet = adv_sampler.sample_for_labeling(vae=vae, discriminator=disc, unlabeled_dataloader=uSet_loader, uSet=uSet)

        else:
            print(f"AL Object | {self.cfg.ACTIVE_LEARNING.SAMPLING_FN} is either not implemented or there is some spelling mistake.")
            raise NotImplementedError(f"AL Object | {self.cfg.ACTIVE_LEARNING.SAMPLING_FN} is either not implemented or there is some spelling mistake.")

        return activeSet, uSet
