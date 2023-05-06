from avalanche.core import SupervisedPlugin
from torch.nn.utils import clip_grad_norm_
from avalanche.training.templates import SupervisedTemplate
from avalanche.core import Template
from loss.multiheadloss import MultiHeadLoss

class LossUpdater(SupervisedPlugin):
    """This plugin updates the MultiHeadLoss's teacher model"""

    def __init__(self, loss: MultiHeadLoss, epochs, grad_clip=False, config=None, schedule = None, decreasing=True, experience=True, step_size=0.5):

        super().__init__()
        self.grad_clip = grad_clip
        self.step_size = step_size
        self.loss = loss
        self.epochs = epochs
        self.config = config
        self.experience = experience
        self.decreasing = decreasing
        if schedule is not None:
            self.schedule = schedule
            self.count = 0
        else:
            self.schedule = None
    
    def update_teacher_model(self, model):
        self.loss.model = model

    def after_forward(self, strategy: "SupervisedTemplate", *args, **kwargs):
        self.loss.update_mb(strategy.mb_x)

    def after_eval_forward(self, strategy: "SupervisedTemplate", *args, **kwargs):
        self.loss.update_mb(strategy.mb_x)

    def after_backward(self, strategy: "SupervisedTemplate", *args, **kwargs):
        if self.grad_clip:
            clip_grad_norm_(strategy.model.parameters(), max_norm=5, norm_type=2.0)

    def before_training(self, strategy: "Template", *args, **kwargs):
        if strategy.kd_rel != 0:
            if self.schedule is not None:
                if self.experience:
                    if self.count % self.schedule == 0:
                        if self.decreasing:
                            strategy.kd_rel -= self.step_size
                        else:
                            strategy.kd_rel += self.step_size
                else:
                    self.count +=1
                    if self.count % self.schedule == 0:
                        if self.decreasing:
                            strategy.kd_rel -= self.step_size
                        else:
                            strategy.kd_rel += self.step_size

    