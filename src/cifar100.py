# Import section
import os
from loss.lossupdater import LossUpdater
from plugins.customTemplate import CustomNaive
from plugins.cutmixplugin import CutMixPlugin
from src.singleheadpretrain import SingleHeadPretrain
from metrics.singleheadaccuracy import SingleHeadAccuracy, TrainedExperienceAccuracy, StreamAccuracy
from metrics.singleheadloss import SingleHeadLoss
from models.customresnets import CustomSlimResnet18, CustomResnet18, CustomResnet32, CustomResnet34
from models.resnet import *
from avalanche.logging import InteractiveLogger, TensorboardLogger, TextLogger
from torch.optim import Adam, AdamW, SGD
from avalanche.benchmarks import SplitCIFAR100
from avalanche.training.supervised import Naive, Replay
from loss.multiheadloss import MultiHeadLoss
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.training.plugins import ReplayPlugin, LRSchedulerPlugin, EarlyStoppingPlugin
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from avalanche.evaluation.metrics import loss_metrics
from avalanche.benchmarks.generators import benchmark_with_validation_stream
from utils.utils import print_dict, save_checkpoint, dict_from_path





def test(training_epochs, kd_rel=5, lr=0.01, loss_type="mse", logdir=None, student_fe="slimres18", pretrain=True,
        temperature=2, cutmix_prob=.5, device_num=None,
        sched_epochs = 0, teacher_type = "resnet34", filename_suffix=None, verbose=False, early_stopping = -1, eval_on_test=True):
    
    """Train the model using Distributed Knowledge Transfer (DKT) using a teacher pretrained on the Joint dataset.

    Args:
        training_epochs (int): Number of epochs
        kd_rel (float): Coefficient of the distillation loss. Defaults to
        lr (float, optional): Learning rate. Defaults to 0.01.
        loss_type (str, optional): Loss function. Defaults to "mse".
        logdir (str, optional): Subdir where the Tensorboard Data will be stored. Defaults to None.
        student_fe (str, optional): Student feature extractor. Defaults to "slimres18".
        pretrain (bool, optional): Sets the pretrain phase for the CL head. Defaults to True.
        temperature (int, optional): Distillation Temperature. Defaults to 2.
        cutmix_prob (float, optional): Sets the target cutmix probability. Defaults to .5.
        device_num (int, optional): Sets the target device. Defaults to None.
        sched_epochs (int, optional): Number of epochs for the StepLR scheduler. Defaults to 0.
        teacher_type (str, optional): Teacher feature extractor. Defaults to "resnet34".
        filename_suffix (_type_, optional): Suffix for the filename. Defaults to None.
        verbose (bool, optional): Enables the verbose training loggger. Defaults to False.
        early_stopping (int, optional): Patience of early stopping. Defaults to -1.
        eval_on_test (bool, optional): If True, no validation set will be used. Defaults to True.

    """

    
    #Teacher path
    TEACHERS_PATH = os.path.join(".", "runs", "Experiment1", "resnet34")
    TEACHER_MODEL = 'resnet34.pth'

    #Student path creation
    student_config = f"student_fe={student_fe}/epochs={training_epochs}/lr={lr}/kd_rels={kd_rel}/temp={temperature}/cm_prob={cutmix_prob}/lr_sched={sched_epochs}/e_stop={early_stopping}"
    config = student_config + f"/teacher_type={teacher_type}"
    config_dict = dict_from_path(student_config)
    print_dict(config_dict, title="Student Config")

    if filename_suffix is None:
        filename_suffix=filename_suffix=student_config.replace('/', '_')

    if logdir is not None:
        TB_PATH = "tb_data/Experiment1/"+ logdir + "/" + filename_suffix
        SAVE_PATH = "runs/Experiment1/" + logdir + "/" + filename_suffix
    else:
        TB_PATH = "tb_data/Experiment1/random_trials/" + filename_suffix
        SAVE_PATH = "runs/Experiment1/random_trials/" + filename_suffix

    if device_num is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
    

    # Create the models
    teacher_model = resnet34()
    teacher_model.load_state_dict(torch.load(os.path.join(TEACHERS_PATH, TEACHER_MODEL)))

    if student_fe == "slimres18":
        model = CustomSlimResnet18()
    else:
        raise ValueError("Model architecture not supported")

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
    if not eval_on_test:
        benchmark = benchmark_with_validation_stream(benchmark_instance=benchmark, validation_size=0.2, shuffle=False)

    # Optimizer
    optimizer = Adam(model.parameters())

    # Load weights on device
    model.to(device)
    teacher_model.to(device)

    if loss_type == "kd":
        loss = MultiHeadLoss(teacher_model, T=temperature, use_mse=False)
    else:
        loss = MultiHeadLoss(teacher_model, T=temperature, use_mse=True)

    # Loggers
    loggers = []
    loggers.append(TensorboardLogger(tb_log_dir=TB_PATH,
                                     filename_suffix=filename_suffix))
    if verbose:
        loggers.append(InteractiveLogger())

    # Evaluation plugin
    eval_plugin = EvaluationPlugin([SingleHeadAccuracy(0, reset_at="epoch", emit_at="epoch", mode="train"),
                                    SingleHeadAccuracy(1, reset_at="epoch", emit_at="epoch", mode="train"),
                                    SingleHeadLoss(0, reset_at="epoch", emit_at="epoch", mode="train"),
                                    SingleHeadLoss(1, reset_at="epoch", emit_at="epoch", mode="train"),
                                    StreamAccuracy(0),
                                    StreamAccuracy(1)],
                                    loss_metrics(epoch=True),
                                    loggers=loggers,
                                    benchmark=benchmark)

    # Plugins
    plugins = []
    plugins.append(ReplayPlugin(2000))
    plugins.append(LossUpdater(loss, training_epochs, grad_clip=True))

    if cutmix_prob != 0:
        plugins.append(CutMixPlugin(beta=1, target_cutmix_prob=cutmix_prob, epochs=training_epochs))
    if sched_epochs != 0:
        plugins.append(LRSchedulerPlugin(scheduler=StepLR(optimizer, sched_epochs, 0.1)))
    if early_stopping != -1:
        plugins.append(EarlyStoppingPlugin(early_stopping, val_stream_name="valid_stream", metric_name="CL stream accuracy/eval_phase/valid_stream/Task000", mode="max"))

    # Init the strategy
    strategy = CustomNaive(model, optimizer, loss, train_mb_size=256, train_epochs=training_epochs, evaluator=eval_plugin,
                        plugins=plugins, device=device, eval_every=early_stopping, kd_rel=kd_rel)


    #Training loop
    for i, exp in enumerate(benchmark.train_stream):
        print(f"Training on experience {i}")

        if eval_on_test:
            train_metrics = strategy.train(exp, eval_streams=[benchmark.test_stream])
        else:
            train_metrics = strategy.train(exp, eval_streams=[benchmark.valid_streams])

        save_checkpoint(strategy, SAVE_PATH, -1, benchmark.get_reproducibility_data())
        print("\n Student metrics:\n")
        print_dict(train_metrics)

    #Evaluation on test set
    if not eval_on_test:
        metrics = strategy.eval(benchmark.test_stream)
        print("\n Metrics on test set \n")
        print_dict(metrics)

    torch.cuda.empty_cache()
    