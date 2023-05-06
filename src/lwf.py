#Import section

from avalanche.training import ParametricBuffer, HerdingSelectionStrategy, Replay
from avalanche.training.templates import SupervisedTemplate
from models.resnet import resnet18
from models.resnet32 import resnet32
from avalanche.models import SlimResNet18
from avalanche.logging import InteractiveLogger, TensorboardLogger, TextLogger
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
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
from torch.optim import SGD, Adam
from models.resnet32 import ResnetClassifier
import torch
import os 

def test(training_epochs, lr=.01, device_num=None, logdir=None, mem_size=2000, temperature=2, optimizer_name="adam", alpha=1,
          verbose=False, filename_suffix=None, early_stopping=False, sched_epochs = 0,  model_fe = "slimres18", eval_on_test = True):
    """Train the model using Distributed Knowledge Transfer (LwF)

    Args:
        training_epochs (int): Number of epochs
        lr (float, optional): Learning rate. Defaults to 0.01.
        device_num (int, optional): Sets the target device. Defaults to None.
        logdir (str, optional): Subdir where the Tensorboard Data will be stored. Defaults to None.
        mem_size (int, optional): Sets the number of patterns to use for the Replay buffer. Defaults to 2000.
        temperature (int, optional): Distillation Temperature. Defaults to 2.
        optimizer_name (str, optional): optimizer type in ["adam", "sgd"]. Defaults to "adam".
        alpha (int, optional): Sets the alpha parameter in LwF.
        verbose (bool, optional): Enables the verbose training loggger. Defaults to False.
        filename_suffix (_type_, optional): Suffix for the filename. Defaults to None.
        early_stopping (int, optional): Patience of early stopping. Defaults to -1.
        sched_epochs (int, optional): Number of epochs for the StepLR scheduler. Defaults to 0.
        model_fe (str, optional): Model feature extractor. Defaults to "slimres18".
        eval_on_test (bool, optional): If True, no validation set will be used. Defaults to True.
    """

    #Model path creation

    config = f"""epochs={training_epochs}/lr={lr}/temp={temperature}/optim={optimizer_name}/
                lr_sched={sched_epochs}/alpha={alpha}/early_stop={early_stopping}"""

    print(f"Model configuration: \n  \
        Model FE: \t {model_fe} \n \
        Epochs: \t{training_epochs} \n \
        Learning Rate: \t{lr} \n \
        Loss: \t Cross Entropy \n \
        Temperature: \t{temperature} \n \
        Replay memory patterns: \t{mem_size} \n")

    if filename_suffix is None:
        filename_suffix=filename_suffix=config.replace('/', '_')

    if logdir is not None:
        TB_PATH = "tb_data/Experiment2/"+ logdir + "/" + filename_suffix
        SAVE_PATH = "runs/" + logdir + "/" + filename_suffix
    else:
        TB_PATH = "tb_data/Experiment2/" + filename_suffix
        SAVE_PATH = "runs/Experiment2/" + filename_suffix
    
    # Load the benchmarkù
    train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.ColorJitter(),
        transforms.Normalize(
            (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        ),
    ])

    benchmark = SplitCIFAR100(10, train_transform=train_transform)
    rep_data = benchmark.get_reproducibility_data()
    if not eval_on_test:
        benchmark = benchmark_with_validation_stream(benchmark_instance=benchmark, validation_size=0.2)

    # Create the model 
    if model_fe == "slimres18":
        model = SlimResNet18(100)
    elif model_fe == "res18":
        model = resnet18()
    elif model_fe == "res32":
        model = resnet32()
    else:
        raise ValueError("Teacher architecture not supported")
    

    #Optimizer
    if optimizer_name == "adam":
        optimizer = Adam(model.parameters(), lr=lr)
    else:
        optimizer = SGD(model.parameters(), lr, 0.9, weight_decay=0.0002)

    # Load weights on device
    if device_num is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")

    model.to(device)
    
    # Loss
    loss = CrossEntropyLoss()

    # Loggers
    loggers = [TensorboardLogger(tb_log_dir=TB_PATH, filename_suffix=filename_suffix)]
    if verbose:
        loggers.append(InteractiveLogger())
    
    # Plugins
    plugins = [ReplayPlugin(mem_size=2000)]
    if sched_epochs != 0:
        plugins.append(LRSchedulerPlugin(StepLR(optimizer, sched_epochs, 0.1)))
    if early_stopping:
        plugins.append(EarlyStoppingPlugin(40, "valid_stream"))
    eval_plugin = EvaluationPlugin(accuracy_metrics(epoch=True, trained_experience=True, stream=True),
                                   loggers=loggers,
                                   benchmark=benchmark)
    
    # Strategy
    strategy = LwF(model, optimizer, loss, alpha=alpha, temperature=temperature, train_mb_size=128,
                     train_epochs=training_epochs, eval_every=20, evaluator=eval_plugin, device=device, plugins=plugins)
    train_model(benchmark, strategy, 0, SAVE_PATH, rep_data, eval_on_test=eval_on_test)


def train_model(benchmark, strategy: SupervisedTemplate, path, rep_data, eval_on_test=True):
    """_summary_

    Args:
        benchmark (_type_): Avalanche Benchmark to train the model.
        strategy (SupervisedTemplate): Avalanhce strategy. 
        path (_type_): Path where the model's checkpoints will be saved.
        rep_data (_type_): Benchmark reproducibility data.
        eval_on_test (bool, optional): If True, no validation set will be used. Defaults to True.

    Returns:
        dict: last record of the metrics
    """
    if eval_on_test:
        eval_stream = benchmark.test_stream
    else:
        eval_stream = benchmark.valid_stream
    
    # Create the save dir
    os.makedirs(path, exist_ok=True)

    for i, exp in enumerate(benchmark.train_stream):
        print(f"Processing experience {i}")
        metrics = strategy.train(exp, eval_streams=[eval_stream])
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
    