# Import section
import torch
from avalanche.training.templates import SupervisedTemplate
from avalanche.core import SupervisedPlugin, Template

def unfreeze_head(strategy):
    strategy.model.student.requires_grad_(True)


def freeze_head(strategy):
    strategy.model.student.requires_grad_(False)


class PretrainPlugin(SupervisedPlugin):
    """Plugin that enables the DKT architecture to pretrain the CL head."""

    def __init__(self, epochs=2):
        super().__init__()
        self.epochs = epochs

    def before_training_exp(self, strategy: Template, *args, **kwargs):
        print("Starting pretraining")
        freeze_head(strategy)
        for _ in range(self.epochs):
            strategy.training_epoch()
        unfreeze_head(strategy)
        print("Stopping pretraining")
