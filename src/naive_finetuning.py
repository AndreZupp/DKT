#Import section
from models.resnet import resnet18
from models.customresnets import CustomSlimResnet18
from plugins.customTemplate import CustomNaive
from metrics import singleheadaccuracy, singleheadloss
from loss.multiheadloss import MultiHeadLoss
from loss.lossupdater import LossUpdater
from torch.nn import CrossEntropyLoss
from avalanche.benchmarks.generators import benchmark_with_validation_stream
from torch.optim import Adam
from torchvision import transforms
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.supervised import Naive
from avalanche.logging import TensorboardLogger, InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.benchmarks import SplitCIFAR100
from avalanche.models import SlimResNet18
from utils.utils import print_dict, save_checkpoint
import torch
import os

def test(training_epochs, lr = 0.01, model_fe="slimres18", logdir=None, device_num=None, filename_suffix=None, sched_epochs=0, verbose=False):
    """Trains the model using the Avalanche Naive Fine-Tuning.

    Args:
        training_epochs (int): Number of epochs
        lr (float, optional): Learning rate. Defaults to 0.01.
        model_fe (str, optional): Model feature extractor. Defaults to "slimres18".
        logdir (str, optional): Subdir where the Tensorboard Data will be stored. Defaults to None.
        device_num (int, optional): Sets the target device. Defaults to None.
        filename_suffix (_type_, optional): Suffix for the filename. Defaults to None.
        sched_epochs (int, optional): Number of epochs for the StepLR scheduler. Defaults to 0.
        verbose (bool, optional): Enables the verbose training loggger. Defaults to False.

    """

    # Model path creation
    student_config = f"Student/epochs={training_epochs}/model={model_fe}/lr={lr}/sched_epochs={sched_epochs}"

    if filename_suffix is None:
        filename_suffix=student_config.replace('/', '_')

    if logdir is not None:
        TB_PATH = "tb_data/nft/"+ logdir + "/" + filename_suffix
        SAVE_PATH = "runs/nft/" + logdir + "/" + filename_suffix
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH, exist_ok=True)
    else:
        TB_PATH = "tb_data/nft/" + filename_suffix
        SAVE_PATH = "runs/nft/" + filename_suffix

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
    benchmark = benchmark_with_validation_stream(benchmark, 0.2, shuffle=False)
    

    if device_num is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")

    # Create the model    
    if model_fe == "slimres18":
        model = SlimResNet18(100)
    elif model_fe == "res18":
        model = resnet18()
    elif model_fe == "customslimres18":
        model = CustomSlimResnet18(100)
    else:
        raise ValueError("Model architecture not supported")
    
    # Loss
    if model_fe == "customslimres18":
        loss = MultiHeadLoss(model, T=2, use_mse=True, no_kd=True)
    else:
        loss = CrossEntropyLoss()

    # Optimizer
    optim = Adam(model.parameters())

    # Loggers
    loggers = []
    loggers.append(TensorboardLogger(tb_log_dir=TB_PATH, filename_suffix=filename_suffix))
    if verbose:
        loggers.append(InteractiveLogger())
    
    # Evaluation Pluigin
    if model_fe != "customslimres18":
        eval_plugin = EvaluationPlugin(accuracy_metrics(stream=True, epoch=True),
                                            loss_metrics(epoch=True),
                                            loggers=loggers,
                                            benchmark=benchmark)
    else:
        eval_plugin = EvaluationPlugin(singleheadaccuracy.SingleHeadAccuracy(0, reset_at="epoch", emit_at="epoch", mode="train"),
                                        singleheadaccuracy.SingleHeadAccuracy(1, reset_at="epoch", emit_at="epoch", mode="train"),
                                        loss_metrics(epoch=True),
                                        singleheadaccuracy.StreamAccuracy(0),
                                        singleheadaccuracy.StreamAccuracy(1),
                                        loggers=loggers,
                                        benchmark=benchmark)
    
    # Plugins and Strategy
    plugins = [ReplayPlugin(mem_size=2000)]
    if model_fe == "customslimres18":
        plugins.append(LossUpdater(loss, 150))
        strategy = CustomNaive(model, optim, loss, 256, training_epochs, device=device, plugins=plugins, evaluator=eval_plugin, eval_every=-1, kd_rel=0)
    else:
        strategy = Naive(model, optim, loss, 256, training_epochs, device=device, plugins=plugins, evaluator=eval_plugin, eval_every=-1)

    # Training loop
    for i, experience in enumerate(benchmark.train_stream):
        print(f"Training on experience {i}")
        metrics = strategy.train(experience, eval_streams=[benchmark.valid_stream])
        save_checkpoint(strategy, path=os.path.join(".","runs", SAVE_PATH), rep_data=rep_data, teacher_number=-1)
        print(f"Metrics of Experience{i}: \n")
        print_dict(metrics)

    #Test results
    test_metrics = strategy.eval(benchmark.test_stream)
    print("Test metrics: \n {}".format(test_metrics))


