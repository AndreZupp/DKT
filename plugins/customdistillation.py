import copy
import torch.nn.functional as F
from avalanche.models import avalanche_forward, MultiTaskModule
from avalanche.training.plugins import LwFPlugin

""" Adapts LwF strategy to perform EEIL distillation """


class CustomDistillation(LwFPlugin):

    def __init__(self, alpha=1, temperature=2, finetuning = False, class_to_distill=None):
        super().__init__(alpha, temperature)
        self.finetuning = finetuning
        self.T = temperature
        self.class_to_distill = class_to_distill
    
    def _distillation_loss(self, out, prev_out, active_units):
        if self.finetuning:
            prev_units = self.class_to_distill
        else:
            prev_units = self.prev_classes['0']
        res = 0
        for elem in prev_units:
            res += F.binary_cross_entropy(F.softmax(out[:, elem] / self.T, dim=0),F.softmax(prev_out[:, elem] / self.T, dim=0))
        return res 
    
    

    def before_backward(self, strategy, **kwargs):
        #strategy.model.classifier.masking = False
        super().before_backward(strategy, **kwargs)
        #strategy.model.classifier.masking = True

    def before_training_exp(self, strategy, **kwargs):
        if self.finetuning:
            """
            Save a copy of the model after each experience and
            update self.prev_classes to include the newly learned classes.
            """
            self.prev_model = copy.deepcopy(strategy.model)
            task_ids = strategy.experience.dataset.task_set
            for task_id in task_ids:
                task_data = strategy.experience.dataset.task_set[task_id]
                pc = set(task_data.targets)

                if task_id not in self.prev_classes:
                    self.prev_classes[str(task_id)] = pc
                else:
                    self.prev_classes[str(task_id)] = self.prev_classes[
                        task_id
                    ].union(pc)
    

    def after_training_exp(self, strategy, **kwargs):
        """
        Save a copy of the model after each experience and
        update self.prev_classes to include the newly learned classes.
        """
        self.prev_model = copy.deepcopy(strategy.model)
        task_ids = strategy.experience.dataset.task_set
        for task_id in task_ids:
            task_data = strategy.experience.dataset.task_set[task_id]
            pc = set(task_data.targets)

            if task_id not in self.prev_classes:
                self.prev_classes[str(task_id)] = pc
            else:
                self.prev_classes[str(task_id)] = self.prev_classes[
                    task_id
                ].union(pc)