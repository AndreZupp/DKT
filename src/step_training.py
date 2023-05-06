# Import section
import os
import torch
import copy
from loss.lossupdater import LossUpdater
from models.customresnets import SlimResNet18
from plugins.customTemplate import CustomNaive
from plugins.cutmixplugin import CutMixPlugin
from metrics.singleheadaccuracy import SingleHeadAccuracy, StreamAccuracy
from metrics.singleheadloss import SingleHeadLoss
from models.customresnets import CustomResnet18, CustomResnet32, CustomSlimResnet18
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.benchmarks import SplitCIFAR100
from loss.multiheadloss import MultiHeadLoss
from avalanche.training.plugins.evaluation import EvaluationPlugin
from torchvision import transforms
from utils.loaders import load_optimizer
from avalanche.benchmarks.generators import benchmark_with_validation_stream
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.training.plugins import ReplayPlugin, EarlyStoppingPlugin, LRSchedulerPlugin
from plugins.model_updater import ModelUpdater
from torch.optim.lr_scheduler import StepLR
from utils.utils import save_checkpoint, print_dict, dict_from_path



def test(training_epochs, optimizer="adam", lr = 0.01, loss_type="mse", kd_rel=0.01, refresh_rate = 10, model_fe="slimres18", logdir=None, 
         device_num=None, filename_suffix=None, kd_scheduler = None, copy_init = True, 
         sched_epochs=0, verbose=False, eval_on_test = True, early_stopping = -1, step_size=0.5):
    
    """Trains two Distributed Knowledge Transfer (DKT) models in the Distributed Continual Learning (DCL) scenario

    Args:
        training_epochs (int): Number of epochs
        optimizer (str, optional): Model optimizer. Defaults to "adam"
        lr (float, optional): Learning rate. Defaults to 0.01.
        loss_type (str, optional): Loss function. Defaults to "mse".
        kd_rel (float): Coefficient of the distillation loss. Defaults to 0.01.
        refresh_rate (int, optional): Sets frequency in model exchange. Defaults to 10.
        model_fe (str, optional): Sets both model feature extractors. Defaults to "slimres18".
        logdir (str, optional): Subdir where the Tensorboard Data will be stored. Defaults to None.
        device_num (int, optional): Sets the target device. Defaults to None.
        filename_suffix (_type_, optional): Suffix for the filename. Defaults to None.
        kd_scheduler (_type_, optional): Sets the kd_rel parameter scheduling. Defaults to None.
        copy_init (bool, optional): uses the same weight init for both models. Defaults to True.
        sched_epochs (int, optional): Number of epochs for the StepLR scheduler. Defaults to 0.
        verbose (bool, optional): Enables the verbose training loggger. Defaults to False.
        eval_on_test (bool, optional): If True, no validation set will be used. Defaults to True.
        early_stopping (int, optional): Patience of early stopping. Defaults to -1.
        step_size (float, optional): The amount at which the kd_rel will be modified by the kd_scheduler. Defaults to 0.5.
    """



    # Student path creation
    student_config = f"model={model_fe}/epochs={training_epochs}/lr={lr}/kd_rels={kd_rel}/early_stop={early_stopping}/lr_sched={sched_epochs}/refresh={refresh_rate}"
    config_dict = dict_from_path(student_config)
    print_dict(config_dict, title="Config")
    
    if filename_suffix is None:
        filename_suffix=filename_suffix=student_config.replace('/', '_')

    if logdir is not None:
        TB_PATH = "tb_data/Experiment3/"+ logdir + "/" + filename_suffix
        SAVE_PATH = "runs/Experiment3/" + logdir + "/" + filename_suffix
    else:
        TB_PATH = "tb_data/Experiment3/Troubleshooting" + "/" + filename_suffix
        SAVE_PATH = "runs/Experiment3/Troubleshooting" +  "/" + filename_suffix

    if device_num is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")

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
    a_benchmark = SplitCIFAR100(10, train_transform=train_transform)
    b_benchmark = SplitCIFAR100(10, train_transform=train_transform)

    a_rep_data = a_benchmark.get_reproducibility_data()
    b_rep_data = b_benchmark.get_reproducibility_data()

    if not eval_on_test:
        a_benchmark = benchmark_with_validation_stream(benchmark_instance=a_benchmark, validation_size=0.2, shuffle=False)
        b_benchmark = benchmark_with_validation_stream(benchmark_instance=b_benchmark, validation_size=0.2, shuffle=False)

    # Models
    if model_fe == "slimres18":
        model_a = CustomSlimResnet18()
        model_b = CustomSlimResnet18()
    elif model_fe == "res32":
        model_a = CustomResnet32()
        model_b = CustomResnet32()
    elif model_fe == "res18":
        model_a = CustomResnet18()
        model_b = CustomResnet18()
    else:
        raise ValueError("Architecture not supported")

    if copy_init is True:
        model_b = copy.deepcopy(model_a)


    # Optimizers 
    a_optim = load_optimizer(model_a, optimizer_name=optimizer)
    b_optim = load_optimizer(model_b, optimizer_name=optimizer)


    # Loss
    if loss_type == 'mse':
        a_loss = MultiHeadLoss(model_b, use_mse=True, multi_head_teacher=True)
        b_loss = MultiHeadLoss(model_a, use_mse=True, multi_head_teacher=True)
    else:
        a_loss = MultiHeadLoss(model_b, use_mse=True, multi_head_teacher=True)
        b_loss = MultiHeadLoss(model_a, use_mse=True, multi_head_teacher=True)

    # Loggers
    a_loggers = []
    b_loggers = []

    a_loggers.append(TensorboardLogger(tb_log_dir=os.path.join(TB_PATH, "model_a")))
    b_loggers.append(TensorboardLogger(tb_log_dir=os.path.join(TB_PATH, "model_b")))

    if verbose:
        a_loggers.append(InteractiveLogger())
        b_loggers.append(InteractiveLogger())

    # Eval plugins
    a_eval_plugin = EvaluationPlugin([SingleHeadAccuracy(0, reset_at="epoch", emit_at="epoch", mode="train", tag="Model A"),
                                        SingleHeadAccuracy(1, reset_at="epoch", emit_at="epoch", mode="train", tag="Model A"),
                                        SingleHeadLoss(0, reset_at="epoch", emit_at="epoch", mode="train", tag="Model A"),
                                        SingleHeadLoss(1, reset_at="epoch", emit_at="epoch", mode="train", tag="Model A"),
                                        StreamAccuracy(0, tag="Model A"),
                                        StreamAccuracy(1, tag="Model A")],
                                        loss_metrics(epoch=True),
                                        loggers=a_loggers,
                                        benchmark=a_benchmark)
    
    b_eval_plugin = EvaluationPlugin([SingleHeadAccuracy(0, reset_at="epoch", emit_at="epoch", mode="train", tag="Model B"),
                                        SingleHeadAccuracy(1, reset_at="epoch", emit_at="epoch", mode="train", tag="Model B"),
                                        SingleHeadLoss(0, reset_at="epoch", emit_at="epoch", mode="train", tag="Model B"),
                                        SingleHeadLoss(1, reset_at="epoch", emit_at="epoch", mode="train", tag="Model B"),
                                        StreamAccuracy(0, tag="Model B"),
                                        StreamAccuracy(1, tag="Model B")],
                                        loss_metrics(epoch=True),
                                        loggers=b_loggers,
                                        benchmark=b_benchmark)
    # Plugins
    a_plugins = []
    b_plugins = []

    a_plugins.append(ReplayPlugin(mem_size=2000))
    a_plugins.append(LossUpdater(a_loss, training_epochs, grad_clip=True, config=logdir, schedule=kd_scheduler, decreasing=False, experience=True, step_size=step_size))
    if sched_epochs != 0:
        a_plugins.append(LRSchedulerPlugin(scheduler=StepLR(a_optim, sched_epochs, 0.1)))
    if early_stopping != -1:
        if eval_on_test:
            a_plugins.append(EarlyStoppingPlugin(early_stopping, val_stream_name="test_stream", metric_name="Model A CL stream accuracy/eval_phase/test_stream/Task000", mode="max", margin=0.01))
        else:
            a_plugins.append(EarlyStoppingPlugin(early_stopping, val_stream_name="test_stream", metric_name="Model A CL stream accuracy/eval_phase/valid_stream/Task000", mode="max", margin=0.01))

    b_plugins.append(ReplayPlugin(mem_size=2000))
    b_plugins.append(LossUpdater(b_loss, training_epochs, grad_clip=True, config=logdir, schedule=kd_scheduler, decreasing=False, experience=True, step_size=step_size))
    if sched_epochs != 0:
        b_plugins.append(LRSchedulerPlugin(scheduler=StepLR(b_optim, sched_epochs, 0.1)))

    if early_stopping != -1:
        if eval_on_test:
            b_plugins.append(EarlyStoppingPlugin(early_stopping, val_stream_name="test_stream", metric_name="Model B CL stream accuracy/eval_phase/test_stream/Task000", mode="max", margin=0.01))
        else:
            b_plugins.append(EarlyStoppingPlugin(early_stopping, val_stream_name="test_stream", metric_name="Model B CL stream accuracy/eval_phase/valid_stream/Task000", mode="max", margin=0.01))
    
    # Send models to device
    model_a.to(device)
    model_b.to(device)

    # Strategies
    a_strategy = CustomNaive(model_a, a_optim, a_loss, train_mb_size=256,
                                train_epochs=training_epochs, evaluator=a_eval_plugin,
                                plugins=a_plugins, device=device, eval_every=early_stopping, kd_rel=kd_rel)
    
    b_strategy = CustomNaive(model_b, b_optim, b_loss, train_mb_size=256,
                                train_epochs=training_epochs, evaluator=b_eval_plugin,
                                plugins=b_plugins, device=device, eval_every=early_stopping, kd_rel=kd_rel)
    
    a_plugins.append(ModelUpdater(a_loss, b_strategy, rate=refresh_rate))
    b_plugins.append(ModelUpdater(b_loss, a_strategy, rate=refresh_rate))

    # Create Save Path
    os.makedirs(os.path.join(SAVE_PATH, "A"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_PATH, "B"), exist_ok=True)


    # Training loop 
    for i in range(10):
        if eval_on_test:
            a_valid_set = [a_benchmark.test_stream]
            b_valid_set = [b_benchmark.test_stream]
        else:
            a_valid_set = [a_benchmark.valid_stream]
            b_valid_set = [b_benchmark.valid_stream]
        print(f"Training model A on experience {i}")
        a_metrics = a_strategy.train(a_benchmark.train_stream[i], eval_streams=a_valid_set)
        
        print(f"Training model B on experience {i}")
        b_metrics = b_strategy.train(b_benchmark.train_stream[i], eval_streams=b_valid_set)


        print(f"Results on Experience {i} \n")
        print("Model A:\n")
        print_dict(a_metrics)
        print("Model B:\n")
        print_dict(b_metrics)

        save_checkpoint(a_strategy, os.path.join(SAVE_PATH,"A"), -1, a_rep_data)
        save_checkpoint(b_strategy, os.path.join(SAVE_PATH,"B"), -1, b_rep_data)

        
    
    if not eval_on_test:
        a_test = a_strategy.eval(a_benchmark.test_stream)
        b_test = b_strategy.eval(b_benchmark.test_stream)
        print("Model A results on test:")
        print_dict(a_test)
        print("Model B results on test:")
        print_dict(b_test)

    torch.cuda.empty_cache()