from avalanche.benchmarks.utils import AvalancheTensorDataset
from avalanche.logging import InteractiveLogger
from avalanche.training import Naive
from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin
from avalanche.training.storage_policy import HerdingSelectionStrategy
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from torch.optim.lr_scheduler import StepLR
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch import cat
from avalanche.benchmarks.generators import nc_benchmark
from plugins.customdistillation import CustomDistillation

""" EEIL (End-to-End Incremental Learning) Finetuning"""


class FineTuning(SupervisedPlugin):

    def __init__(self, mem_size, epochs, num_classes=100, custom_start=0):
        super(FineTuning, self).__init__()
        self.seen_classes = set()
        self.sample_per_class = int(mem_size / num_classes)
        self.epochs = epochs
        self.sampled_x = None  # Saves the indices of the pattern
        self.sampled_y = None
        self.custom_start = custom_start

    def after_training_exp(self, strategy: SupervisedTemplate, *args, **kwargs):
        class_balanced_indices = []
        selection_policy = HerdingSelectionStrategy(model=strategy.model, layer_name="resnet_model")
        current_exp_classes = strategy.adapted_dataset[:][1].tolist()
        new_classes = set(current_exp_classes) - self.seen_classes
        new_classes_indices = dict(zip(new_classes, [[] for x in new_classes]))

        for position, class_id in enumerate(current_exp_classes):
            if class_id in new_classes:
                new_classes_indices[class_id].append(position)
        
        for new_class in new_classes:
            data = strategy.adapted_dataset[new_classes_indices[new_class]]
            x = AvalancheTensorDataset(data[0], data[1])
            class_balanced_indices.extend(selection_policy.make_sorted_indices(strategy, x)[:self.sample_per_class])

        new_data = strategy.adapted_dataset[class_balanced_indices]

        if self.sampled_x is None:
            self.sampled_x = new_data[0]
            self.sampled_y = new_data[1]
        else:
            self.sampled_x = cat((self.sampled_x, new_data[0]), dim=0)
            self.sampled_y = cat((self.sampled_y, new_data[1]), dim=0)
    

        new_dataset = AvalancheTensorDataset(self.sampled_x, self.sampled_y)
        self.seen_classes.update(new_classes)

        if strategy.experience.current_experience <= self.custom_start or not strategy.is_training:
            return
            
        loss = CrossEntropyLoss()
        optim = SGD(strategy.model.parameters(), lr=0.01)
        print(f"New dataset has {self.sampled_x.shape} patterns")
        benchmark = nc_benchmark(train_dataset=new_dataset, test_dataset=new_dataset, n_experiences=1,
                                 task_labels=False)
        lr_scheduler = LRSchedulerPlugin(scheduler=StepLR(optim, 10, 0.1))
        distillation = CustomDistillation(finetuning=True, class_to_distill=strategy.experience.classes_in_this_experience)
        ft_strategy = Naive(strategy.model, optim, loss, train_mb_size=128, train_epochs=self.epochs,
                            device=strategy.device, plugins=[lr_scheduler, distillation],
                            evaluator=EvaluationPlugin(benchmark=benchmark))
        print("Start Fine Tuning")
        ft_strategy.train(benchmark.train_stream)
        print("End Fine Tuning")
