from avalanche.core import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate
import torch
import warnings
import numpy as np


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class CutMixPlugin(SupervisedPlugin):

    def __init__(self, beta, epochs, target_cutmix_prob=0.1, loss=None):
        super(CutMixPlugin, self).__init__()
        self.beta = beta
        self.loss = loss
        self.epochs = epochs
        self.target_cutmix_prob = target_cutmix_prob

    def before_training_exp(self, strategy: SupervisedTemplate, *args, **kwargs):
        self.cutmix_prob = 0.1

    def before_training_iteration(self, strategy: SupervisedTemplate, *args, **kwargs):
        r = np.random.rand(1)
        if self.beta > 0 and r < self.cutmix_prob:
            lam = np.random.beta(self.beta, self.beta)
            index = torch.randperm(strategy.mbatch[0].shape[0]).cuda()
            target_b = strategy.mbatch[1][index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(strategy.mbatch[0].shape, lam)
            strategy.mbatch[0][:, :, bbx1:bbx2, bby1:bby2] = strategy.mbatch[0][index, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (strategy.mbatch[0].shape[-1] * strategy.mbatch[0].shape[-2]))
            self.cutmix_targets = target_b
            self.lam = lam
            self.cutmix_mb = True
        else:
            self.cutmix_mb = False

    def after_training_epoch(self, strategy: SupervisedTemplate, *args, **kwargs):
        self.cutmix_prob += (self.target_cutmix_prob - 0.1 / self.epochs)

