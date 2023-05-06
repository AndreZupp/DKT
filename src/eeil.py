# Import section
import os

from avalanche.training import ParametricBuffer, HerdingSelectionStrategy, Replay
from avalanche.training.templates import SupervisedTemplate
from models.resnet import *
from avalanche.logging import InteractiveLogger, TensorboardLogger, TextLogger
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from torch.optim.lr_scheduler import StepLR
from avalanche.benchmarks import SplitCIFAR100
from avalanche.training.supervised import LwF, Naive
from avalanche.training.plugins.evaluation import EvaluationPlugin
from torchvision import transforms
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.benchmarks.generators import benchmark_with_validation_stream
from torch.nn import CrossEntropyLoss
from utils.loaders import load_optimizer
from utils.best_teacher_configs import best_configs
from models.multiheadresnet34 import MultiHeadResnet
from plugins.finetuning import FineTuning
from plugins.gradientnoise import GradientNoise
from plugins.customdistillation import CustomDistillation
from avalanche.training.plugins import EarlyStoppingPlugin
from torch.optim import SGD

""" Avalanche implementation of End-to-End Incremental Learning (EEIL)"""


def test(training_epochs=40, optimizer_name="sgd", mem_size=2000, lr=0.01, logdir=None, device_num=None, filename_suffix=None, finetuning=True,
         earlystopping=True, momentum = 0.9, scheduler_epochs = 10, finetuning_epochs = 30, temperature = 2,
         verbose = False, strategy_type = "eeil", train=True):
    """_summary_

    Args:
        training_epochs (int): Number of epochs. Defaults to 40.
        optimizer_name (str, optional): Model . Defaults to "sgd".
        mem_size (int, optional): Sets the number of patterns stored in the Replay memory. Defaults to 2000.
        lr (float, optional): Learning rate. Defaults to 0.01.
        logdir (str, optional): Subdir where the Tensorboard Data will be stored. Defaults to None.
        device_num (int, optional): Sets the target device. Defaults to None.
        filename_suffix (_type_, optional): Suffix for the filename. Defaults to None.
        finetuning (bool, optional): Enables Finetuning. Defaults to True.
        earlystopping (bool, optional): Enables Earlystopping. Defaults to True.
        momentum (float, optional): Sets the momentum parameter. Defaults to 0.9.
        sched_epochs (int, optional): Number of epochs for the StepLR scheduler. Defaults to 10.
        finetuning_epochs (int, optional): _description_. Defaults to 30.
        temperature (int, optional): Distillation Temperature. Defaults to 2.
        verbose (bool, optional): Enables the verbose training loggger. Defaults to False.
        strategy_type (str, optional): Switches between LwF and EEIL. Defaults to "eeil".
        train (bool, optional): _description_. Defaults to True.

    Returns:
        Returns the strategy's last recorded metrics.
    """


    config = f"epochs={training_epochs}/lr={lr}/mem_size={mem_size}/sched_epoch={scheduler_epochs}/ft_epoch={finetuning_epochs}/temp={temperature}/optim={optimizer_name}/mom={momentum}"
    print(config)
    if filename_suffix is None:
        filename_suffix=filename_suffix=config.replace('/', '_')
    if logdir is not None:
        TB_PATH = "tb_data/"+ logdir + "/" + filename_suffix
        SAVE_PATH = "runs/" + logdir + "/" + filename_suffix
    else:
        TB_PATH = "tb_data/EEIL/" + filename_suffix
        SAVE_PATH = "runs/EEIL/" + filename_suffix
    


    # Create the models
    teacher_model = MultiHeadResnet()

    # Load the benchmark
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.ColorJitter(),
            transforms.Normalize(
                (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
            ),
        ]
    )
    benchmark = SplitCIFAR100(10, train_transform=train_transform)
    rep_data = benchmark.get_reproducibility_data()
    if train:
        benchmark = benchmark_with_validation_stream(benchmark_instance=benchmark, validation_size=0.2)

    # Training parameters
    if optimizer_name == "adam":
        optimizer = load_optimizer(model=teacher_model, optimizer_name=optimizer_name, lr=lr)
    else:
        optimizer = SGD(teacher_model.parameters(),lr=lr, momentum=momentum, weight_decay=0.0001)

    # Load model weights on device
    if device_num is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")

    teacher_model.to(device)
    loss = CrossEntropyLoss()

    # Loggers
    loggers = [TensorboardLogger(tb_log_dir=TB_PATH, filename_suffix=filename_suffix)]
    if verbose:
        loggers.append(InteractiveLogger())

    # Plugins
    lr_scheduler = LRSchedulerPlugin(scheduler=StepLR(optimizer, scheduler_epochs, 0.1))
    plugins = []
    if finetuning:
        plugins.append(FineTuning(epochs=finetuning_epochs, mem_size=mem_size, custom_start=2))
    if earlystopping:
        plugins.append(EarlyStoppingPlugin(30, "valid_stream"))
    policy = ParametricBuffer(mem_size, groupby="class",
                              selection_strategy=HerdingSelectionStrategy(teacher_model, "resnet_model"))
    plugins.append(ReplayPlugin(mem_size, storage_policy=policy))
    plugins.append(lr_scheduler)
    plugins.append(GradientNoise(eta= 0.3, gamma = 0.55))

    # Evaluation plugin
    eval_plugin = EvaluationPlugin(accuracy_metrics(epoch=True, trained_experience=True, stream=True),
                                   loggers=loggers,
                                   benchmark=benchmark)
    if strategy_type == "eeil":
        plugins.append(CustomDistillation(temperature=temperature))
        strategy =  Naive(teacher_model, optimizer, loss, train_mb_size=128, train_epochs=training_epochs,
                   evaluator=eval_plugin,
                   device=device, eval_every=10, plugins=plugins)
    else:
        strategy = LwF(teacher_model, optimizer, loss, alpha=1, temperature=temperature, train_mb_size=256,
                     train_epochs=training_epochs, eval_every=10, evaluator=eval_plugin, device=device, plugins=plugins )
    
    
    metrics = train_model(benchmark, strategy, 0, path=SAVE_PATH, rep_data= rep_data)

    if train:
        test_results = strategy.eval(benchmark.test_stream)
        print("Metrics on test set:\n", test_results)
    else:
        print("Metrics on test set:\n", metrics)
    return metrics


def train_model(benchmark, strategy: SupervisedTemplate, step, path, rep_data, train=True):
    os.makedirs(path, exist_ok=True)
    if train:
        for i in range(step, len(benchmark.train_stream)):
            print(f"Processing experience {i}")
            strategy.model.adaptation(benchmark.train_stream[i])
            metrics = strategy.train(benchmark.train_stream[i], eval_streams=[benchmark.valid_stream])
            print(f"Metrics of experience{i}: \n", metrics)
            save_checkpoint(strategy, path, rep_data)
    else:
        for i in range(step, len(benchmark.train_stream)):
            strategy.model.adaptation(benchmark.train_stream[i])
            metrics = strategy.train(benchmark.train_stream[i], eval_streams=[benchmark.test_stream])
            save_checkpoint(strategy, path, rep_data)
    torch.cuda.empty_cache()
    return metrics



def save_checkpoint(strategy, path, rep_data):
    torch.save({
        'experience': strategy.experience.current_experience,
        'model_state_dict': strategy.model.state_dict(),
        'optimizer_state_dict': strategy.optimizer.state_dict(),
        'loss': strategy.loss,
        'rep_data': rep_data,
    }, f"{path}/Experience{strategy.experience.current_experience}.pt")

