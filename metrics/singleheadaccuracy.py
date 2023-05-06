from avalanche.evaluation.metrics.accuracy import AccuracyPluginMetric
from avalanche.training.templates import SupervisedTemplate

class SingleHeadAccuracy(AccuracyPluginMetric):
    """Adapts the AccuracyPlugin to work with the Distributed Knowledge Transfer (DKT) architecture."""

    def __init__(self, head, reset_at, emit_at, mode, tag=""):
        super(SingleHeadAccuracy, self).__init__(reset_at, emit_at, mode)
        self.head = head
        self.mode = mode
        self.tag = tag 

    def update(self, strategy: "SupervidedTemplate"):
        if hasattr(strategy.experience, "task_labels"):
            task_labels = strategy.experience.task_labels
        else:
            task_labels = [0]

        if len(task_labels) > 1:
            task_labels = strategy.mb_task_id
        else:
            task_labels = task_labels[0]
        self._accuracy.update(strategy.mb_output[self.head], strategy.mb_y, task_labels)

    def __str__(self):
        if self.head == 0:
            return f"{self.tag} CL {self.mode} accuracy"
        else:
            return f"{self.tag} ST {self.mode} accuracy"

class StreamAccuracy(SingleHeadAccuracy):
    """
    At the end of the entire stream of experiences, this plugin metric
    reports the average accuracy over all patterns seen in all experiences.
    This metric only works at eval time.
    """

    def __init__(self, head, tag=""):
        """
        Creates an instance of StreamAccuracy metric
        """
        super(StreamAccuracy, self).__init__(head,
            reset_at="stream", emit_at="stream", mode="eval", tag=tag)
        self.head = head
        self._current_experience = 0 

    def __str__(self):
        if self.head == 0:
            return f"{self.tag} CL stream accuracy"
        else:
            return f"{self.tag} ST stream accuracy"

class TrainedExperienceAccuracy(SingleHeadAccuracy):
    """
    At the end of each experience, this plugin metric reports the average
    accuracy for only the experiences that the model has been trained on so far.

    This metric only works at eval time.
    """

    def __init__(self, head):
        """
        Creates an instance of TrainedExperienceAccuracy metric by first
        constructing AccuracyPluginMetric
        """
        super(TrainedExperienceAccuracy, self).__init__(head, reset_at="stream",
                                                        emit_at="stream", mode="eval")
        self.head = head
        self._current_experience = 0

    def after_training_exp(self, strategy) -> None:
        self._current_experience = strategy.experience.current_experience
        # Reset average after learning from a new experience
        SingleHeadAccuracy.reset(self, strategy)
        return SingleHeadAccuracy.after_training_exp(self, strategy)

    def update(self, strategy):
        """
        Only update the accuracy with results from experiences that have been
        trained on
        """
        if strategy.experience.current_experience <= self._current_experience:
            SingleHeadAccuracy.update(self, strategy)

    def __str__(self):
        if self.head == 0:
            return f"{self.tag} CL Trained exp accuracy"
        else:
            return f"{self.tag} Student Trained exp accuracy"
