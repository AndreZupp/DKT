from avalanche.evaluation.metrics.loss import LossPluginMetric

class SingleHeadLoss(LossPluginMetric):
    """ Adapts the Avalanche LossPluginMetric to work with the Distributed Knowledge Transfer (DKT) architecture """

    def __init__(self, head, reset_at, emit_at, mode, tag=""):
        super().__init__(reset_at, emit_at, mode)
        self.head = head
        self.tag = tag
    
    def update(self, strategy):
        # task labels defined for each experience
        if hasattr(strategy.experience, "task_labels"):
            task_labels = strategy.experience.task_labels
        else:
            task_labels = [0]  # add fixed task label if not available.

        if len(task_labels) > 1:
            # task labels defined for each pattern
            # fall back to single task case
            task_label = 0
        else:
            task_label = task_labels[0]
        self._loss.update(
            strategy.criterion()[self.head], patterns=len(strategy.mb_y), task_label=task_label
        )

    def __str__(self) -> str:
        if self.head == 0:
            return f"{self.tag} CL head loss"
        else:
            return f"{self.tag} ST head loss"

    