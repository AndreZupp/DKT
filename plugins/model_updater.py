from avalanche.core import SupervisedPlugin, Template
from loss.multiheadloss import MultiHeadLoss


class ModelUpdater(SupervisedPlugin):

    def __init__(self, loss: MultiHeadLoss, target_strategy, rate):
        super(ModelUpdater, self).__init__()
        self.target_strategy = target_strategy
        self.rate = rate
        self.counter = 1
        self.loss = loss

    def update_model(self):
        self.loss.model = self.target_strategy.model

    def before_training_exp(self, strategy: Template, *args, **kwargs):
        self.update_model()
        self.counter = 1

    def before_training_epoch(self, strategy: Template, *args, **kwargs):
        if self.counter % self.rate == 0:
            self.update_model()
        self.counter += 1