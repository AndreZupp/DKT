# Import section

import torch
import os
from models.customresnets import CustomResnet18, CustomResnet32, CustomSlimResnet18
from models.resnet import resnet18
from avalanche.models import SlimResNet18
from loss.lossupdater import LossUpdater
from plugins.cutmixplugin import CutMixPlugin
from metrics.singleheadaccuracy import SingleHeadAccuracy, StreamAccuracy
from metrics.singleheadloss import SingleHeadLoss
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.benchmarks import SplitCIFAR100
from loss.multiheadloss import MultiHeadLoss
from avalanche.training.plugins.evaluation import EvaluationPlugin
from torchvision import transforms
from utils.loaders import load_optimizer
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.training.plugins import ReplayPlugin, EarlyStoppingPlugin, LRSchedulerPlugin
from models.resnet32 import resnet32
from torch.optim.lr_scheduler import StepLR
from utils.utils import print_dict, dict_from_path
from plugins.customTemplate import CustomNaive


# # # The script is uncomplete. Please refer to the file "splicifar100_pretrained" to use the version with a pretrained teacher # # #

def test(training_epochs, teacher_strategy, teacher_type, optimizer="adam", lr = 0.01, loss="mse", kd_rel=5, temperature=1, mem_size = 2000, logdir=None,
         cutmix_prob=0, device_num=None, filename_suffix=None, kd_scheduler = None,
         sched_epochs=0, verbose=False, eval_on_test = True, student_fe="slimres18"):


    # Student path creation
    student_config = f"""student_fe={student_fe}/teacher_type={teacher_type}/epochs={training_epochs}/lr={lr}
                        /loss={loss}/kd_rels={kd_rel}/temp={temperature}/cm_prob={cutmix_prob}/lr_sched={sched_epochs}"""
    config_dict = dict_from_path(student_config)
    print_dict(config_dict, title="Student Configuration")
    
    if filename_suffix is None:
        filename_suffix=student_config.replace('/', '_')

    if logdir is not None:
        TB_PATH = "tb_data/Experiment2/"+ logdir + "/" + filename_suffix
        SAVE_PATH = "runs/Experiment2/" + logdir + "/" + filename_suffix
    else:
        TB_PATH = "tb_data/random_trials/" + filename_suffix
        SAVE_PATH = "runs/random_trials/" + filename_suffix

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
    benchmark = SplitCIFAR100(10, train_transform=train_transform)

    # Student model
    if student_fe == "slimres18":
        student_model = CustomSlimResnet18()
    elif student_fe == "res32":
        student_model = CustomResnet32()
    elif student_fe == "res18":
        student_model = CustomResnet18()
    else:
        raise ValueError("Student architecture not supported")
    student_optimizer = load_optimizer(student_model, optimizer_name=optimizer)
    student_model.to(device)
    
    # Loss
    if loss == 'mse':
        student_loss = MultiHeadLoss(teacher_strategy.model, kd_rel, use_mse=True)
    else:
        student_loss = MultiHeadLoss(teacher_strategy.model, kd_rel, temperature)
    
    # Loggers
    student_loggers = []
    student_loggers.append(TensorboardLogger(tb_log_dir=os.path.join(TB_PATH,"student")))
    if verbose:
        student_loggers.append(InteractiveLogger())
    
    # Evaluation plugin
    student_eval_plugin = EvaluationPlugin([SingleHeadAccuracy(0, reset_at="epoch", emit_at="epoch", mode="train"),
                                            SingleHeadAccuracy(1, reset_at="epoch", emit_at="epoch", mode="train"),
                                            SingleHeadLoss(0, reset_at="epoch", emit_at="epoch", mode="train"),
                                            SingleHeadLoss(1, reset_at="epoch", emit_at="epoch", mode="train"),
                                            StreamAccuracy(0),
                                            StreamAccuracy(1)],
                                           loss_metrics(epoch=True),
                                           loggers=student_loggers,
                                           benchmark=benchmark)
    
    # Teacher evaluation plugin
    teacher_eval_plugin = EvaluationPlugin(accuracy_metrics(epoch=True, stream=True),
                                           loss_metrics(epoch=True),
                                           loggers=[TensorboardLogger(tb_log_dir=os.path.join(TB_PATH, "teacher"))],
                                           benchmark=benchmark)
    # Teacher strategy
    teacher_strategy.evaluator = teacher_eval_plugin

    # Student plugins
    student_plugins = []
    student_plugins.append(LossUpdater(student_loss, training_epochs, grad_clip=True, config=logdir, schedule=kd_scheduler))
    if sched_epochs != 0:
        student_plugins.append(LRSchedulerPlugin(scheduler=StepLR(student_optimizer, sched_epochs, 0.1)))
    if cutmix_prob != 0:
        student_plugins.append(CutMixPlugin(beta = 1, target_cutmix_prob=cutmix_prob, epochs=training_epochs))
    
    student_plugins.append(ReplayPlugin(mem_size=mem_size))
    student_strategy = CustomNaive(student_model, student_optimizer, student_loss, train_mb_size=256,
                              train_epochs=training_epochs, evaluator=student_eval_plugin,
                              plugins=student_plugins, device=device, eval_every=-1)

    # Train loop
    if maintain_class_order:
        metrics = train_model(benchmark, teacher_strategy, student_strategy, eval_on_test)

    return metrics



def train_model(benchmark, teacher_strategy, student_strategy, eval_on_test):
    for i in range(0, len(benchmark.train_stream)):
        print(f"Training the teacher on experience {i}")
        teacher_strategy.train(benchmark.train_stream[i])
        student_strategy.plugins[0].update_teacher_model(teacher_strategy.model)
        print(f"Training the student on experience {i}")
        student_strategy.train(benchmark.train_stream[i])
        if eval_on_test:
            student_metrics = student_strategy.eval(benchmark.test_stream)
            teacher_metrics = teacher_strategy.eval(benchmark.test_stream)
        else:
            student_metrics = student_strategy.eval(benchmark.valid_stream)
            teacher_metrics = teacher_strategy.eval(benchmark.valid_stream)
        print("\n Teacher metrics:\n")
        print_dict(teacher_metrics)
        print("\n Student metrics:\n")
        print_dict(student_metrics)
        
    torch.cuda.empty_cache()
    return (student_metrics, teacher_metrics)


def save_checkpoint(strategy, path, teacher_number):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    torch.save({
        'experience': strategy.experience.current_experience,
        'model_state_dict': strategy.model.state_dict(),
        'optimizer_state_dict': strategy.optimizer.state_dict(),
        'loss': strategy.loss,
        'teacher_number': teacher_number,
    }, f"{path}/Experience{strategy.experience.current_experience}.pt")