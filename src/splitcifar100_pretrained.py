# Import section
import os
from loss.lossupdater import LossUpdater
from models.customresnets import SlimResNet18
from plugins.customTemplate import CustomNaive
from plugins.cutmixplugin import CutMixPlugin
from metrics.singleheadaccuracy import SingleHeadAccuracy, StreamAccuracy
from metrics.singleheadloss import SingleHeadLoss
from models.customresnets import CustomResnet18, CustomResnet32, CustomSlimResnet18
from models.multiheadresnet34 import MultiHeadResnet
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.models import SlimResNet18
from torch.optim import Adam, SGD
from avalanche.benchmarks import SplitCIFAR100
from avalanche.training.supervised import Naive
from loss.multiheadloss import MultiHeadLoss
from avalanche.training.plugins.evaluation import EvaluationPlugin
from torchvision import transforms
from utils.loaders import load_optimizer
from avalanche.benchmarks.generators import benchmark_with_validation_stream
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.training.plugins import ReplayPlugin, EarlyStoppingPlugin, LRSchedulerPlugin
import torch
from models.resnet32 import ResnetClassifier
from torch.optim.lr_scheduler import StepLR
from utils.loaders import load_teacher_classes, load_teacher_checkpoints
from utils.utils import save_checkpoint, print_dict, dict_from_path





def test(training_epochs, optimizer="adam", lr = 0.01, loss_type="mse", kd_rel=.5, temperature=2, mem_size = 2000, logdir=None,
         cutmix_prob=0, device_num=None, filename_suffix=None, teacher_type="lwf_slimres18", teacher_config = 1, kd_scheduler = None,
          maintain_class_order=False, sched_epochs=0, verbose=False, eval_on_test = True, early_stopping = -1,
          student_fe = "slimres18"):
    
    """Train the model using Distributed Knowledge Transfer (DKT) using a pre-trained Continual Learning Teacher.

    Args:
        training_epochs (int): Number of epochs
        optimizer (str, optional): Model . Defaults to "adam".
        lr (float, optional): Learning rate. Defaults to 0.01.
        loss_type (str, optional): Loss function. Defaults to "mse".
        kd_rel (float): Coefficient of the distillation loss. Defaults to 0.5.
        temperature (int, optional): Distillation Temperature. Defaults to 2.
        mem_size (int, optional): Sets the number of patterns stored in the Replay memory. Defaults to 2000.
        logdir (str, optional): Subdir where the Tensorboard Data will be stored. Defaults to None.
        cutmix_prob (float, optional): Sets the target cutmix probability. Defaults to .5.
        device_num (int, optional): Sets the target device. Defaults to None.
        filename_suffix (_type_, optional): Suffix for the filename. Defaults to None.
        teacher_type (str, optional): Sets the teacher feature extractor. Defaults to "lwf_slimres18".
        teacher_config (int, optional): Takes the specified teacher number. Defaults to 1.
        kd_scheduler (_type_, optional): Sets the kd_rel scheduler. Defaults to None.
        maintain_class_order (bool, optional): If True teacher and student will have the same class order. Defaults to False.
        sched_epochs (int, optional): Number of epochs for the StepLR scheduler. Defaults to 0.
        verbose (bool, optional): Enables the verbose training loggger. Defaults to False.
        eval_on_test (bool, optional): If True, no validation set will be used. Defaults to True.
        early_stopping (int, optional): Patience of early stopping. Defaults to -1.
        student_fe (str, optional): Sets the student feature extractor. Defaults to "slimres18".
    """

    # Student path creation
    student_config = f"teacher={teacher_type}/student={student_fe}/epochs={training_epochs}/lr={lr}/kd_rels={kd_rel}/temp={temperature}/cm_prob={cutmix_prob}/lr_sched={sched_epochs}"
    config = student_config + f"/teacher_type={teacher_type}/teacher_number={teacher_config}"
    config_dict = dict_from_path(config)
    print_dict(config_dict, title="Student Config")


    if filename_suffix is None:
        filename_suffix=filename_suffix=student_config.replace('/', '_')

    if logdir is not None:
        TB_PATH = "tb_data/Experiment2/"+ logdir + "/" + filename_suffix
        SAVE_PATH = "runs/Experiment2/" + logdir + "/" + filename_suffix
    else:
        TB_PATH = "tb_data/Experiment2/random_trials/" + filename_suffix
        SAVE_PATH = "runs/Experiment2/random_trials/" + filename_suffix


    if device_num is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    if not maintain_class_order:
        benchmark = SplitCIFAR100(10, train_transform=train_transform)
    else:
        class_order = load_teacher_classes(teacher_type, teacher_config, device=device)
        benchmark = SplitCIFAR100(10, train_transform=train_transform, fixed_class_order=class_order)
    
    if not eval_on_test:
        benchmark = benchmark_with_validation_stream(benchmark_instance=benchmark, validation_size=0.2, shuffle=False)

    # Load teacher checkpoints
    teacher_checkpoints = load_teacher_checkpoints(teacher_type, teacher_config, device=device)


    # Student Model
    if student_fe == "slimres18":
        student_model = CustomSlimResnet18()
    elif student_fe == "res32":
        student_model = CustomResnet32()
    elif student_fe == "res18":
        student_model = CustomResnet18()
    else:
        raise ValueError("Student architecture not supported")
        
    # Optimizer
    student_optimizer = load_optimizer(student_model, optimizer_name=optimizer)

    # Loss
    if loss_type == 'mse':
        student_loss = MultiHeadLoss(None, use_mse=True)
    else:
        student_loss = MultiHeadLoss(None, temperature)


    # Loggers
    loggers = []
    loggers.append(TensorboardLogger(tb_log_dir=os.path.join(TB_PATH, "student")))
    
    if verbose:
        loggers.append(InteractiveLogger())

    # Student evaluation plugin
    student_eval_plugin = EvaluationPlugin([SingleHeadAccuracy(0, reset_at="epoch", emit_at="epoch", mode="train"),
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
    plugins.append(LossUpdater(student_loss, training_epochs, grad_clip=True, config=logdir, schedule=kd_scheduler))
    plugins.append(ReplayPlugin(mem_size=mem_size))
    if sched_epochs != 0:
        plugins.append(LRSchedulerPlugin(scheduler=StepLR(student_optimizer, sched_epochs, 0.1)))
    if cutmix_prob != 0:
        plugins.append(CutMixPlugin(beta = 1, target_cutmix_prob=cutmix_prob, epochs=training_epochs))
    if early_stopping != -1:
        plugins.append(EarlyStoppingPlugin(early_stopping, val_stream_name="test_stream", metric_name="CL stream accuracy/eval_phase/test_stream/Task000", mode="max", margin=0.03))
    
    # Strategy
    student_model.to(device)

    student_strategy = CustomNaive(student_model, student_optimizer, student_loss, train_mb_size=256,
                                train_epochs=training_epochs, evaluator=student_eval_plugin,
                                plugins=plugins, device=device, eval_every=early_stopping, kd_rel=kd_rel) 

    # Teacher model
    if teacher_type == "lwf_res32":
        teacher_model = ResnetClassifier(100)
    elif teacher_type == "lwf_slimres18":
        teacher_model = SlimResNet18(100)
    elif teacher_type == "eeil_res32":
        teacher_model = MultiHeadResnet()
    else:
        raise ValueError("Teacher architecture not supported")
    
    # Teacher section 
    teacher_model.to(device)
    os.makedirs(os.path.join(TB_PATH, "teacher"), exist_ok=True)
    teacher_loggers = [TensorboardLogger(os.path.join(TB_PATH, "teacher"))]
    teacher_eval_plugin = EvaluationPlugin(accuracy_metrics(stream=True), benchmark=benchmark, loggers=teacher_loggers)
    teacher_strategy = Naive(teacher_model, Adam(teacher_model.parameters()), evaluator=teacher_eval_plugin, device=device)

    # Training loop
    for i, exp in enumerate(benchmark.train_stream):
        print(f"Training on experience: {i}")
        teacher_model.load_state_dict(teacher_checkpoints[i])

        #Send teacher to student 
        student_strategy.plugins[0].update_teacher_model(teacher_model)

        if not eval_on_test:
            student_metrics = student_strategy.train(exp, eval_streams=[benchmark.valid_stream])
            teacher_metris = teacher_strategy.eval(benchmark.valid_stream) 
        else:
            student_metrics = student_strategy.train(exp, eval_streams=[benchmark.test_stream])
            teacher_metris = teacher_strategy.eval(benchmark.test_stream)
            
        save_checkpoint(student_strategy, SAVE_PATH, teacher_config, benchmark.get_reproducibility_data())
        print("\n Teacher metrics:\n")
        print_dict(teacher_metris)
        print("\n Student metrics:\n")
        print_dict(student_metrics)
        
    torch.cuda.empty_cache()

    