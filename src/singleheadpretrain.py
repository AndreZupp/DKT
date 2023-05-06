import warnings
warnings.simplefilter("ignore", UserWarning)
from avalanche.training import Naive


class SingleHeadPretrain():

    def __init__(self, strategy: Naive):
        self.strategy = strategy

    def freeze_head(self):
        self.strategy.model.student.requires_grad_(False)

    def unfreeze_head(self):
        self.strategy.model.student.requires_grad_(True)

    def pretrain(self, experience):
        self.freeze_head()
        print("Start of the pretraining")
        self.strategy.train(experience)
        print("End of the pretraining")
        self.unfreeze_head()