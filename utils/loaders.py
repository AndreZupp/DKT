from torch.optim import Adam, SGD, AdamW
from torch.nn import MSELoss, CrossEntropyLoss
from loss.multiheadloss import MultiHeadLoss
from models.customresnets import CustomSlimResnet18, CustomResnet18, CustomResnet32
from models.customresnet34 import CustomResnet34
from models.resnet import resnet34
from utils.best_teacher_configs import best_eeil_configs, best_lwf_res32_configs, best_lwf_slimres18_configs
import torch
import os

def load_optimizer(model, config=None, optimizer_name=None, lr=0.01):
    if config is not None:
        optimizer_name = config['optimizer']
    if optimizer_name == 'adam':
        return Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        return SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'adamw':
        return AdamW(model.parameters(), lr=lr)
    else:
        raise ValueError("Unknown Optimizer name. Valid choices are: adam, sgd, adamw")




def load_loss(config, teacher_model):
    loss_type = config['loss']

    if loss_type == 'mse':
        return MSELoss()
    elif loss_type == 'student_mse':
        return MultiHeadLoss(teacher_model, config['kd_rel'], use_mse=True)
    elif loss_type == 'student_kd':
        return MultiHeadLoss(teacher_model, config['kd_rel'], T=config['T'])
    elif loss_type == 'crossentropy':
        return CrossEntropyLoss()
    else:
        raise ValueError("Loss type not supported. Valid choices are: mse, crossentropy, student_mse, student_kd")


def load_model(config):
    model = config['model']
    if model == "customslimresnet18":
        return CustomSlimResnet18()
    elif model == "customresnet34":
        return CustomResnet34()
    elif model == "resnet34":
        return resnet34()
    else:
        raise ValueError("Model not supported. Valid choices are: customslimresnet18, customresnet34, resnet34")

def load_teacher_classes(teacher_type, teacher_config, device, number_of_experiences = 10):

    if teacher_type == "lwf_res32":
        base_path = "runs/Experiment2/lwf_res32"
        path = best_lwf_res32_configs[teacher_config]
    elif teacher_type == "lwf_slimres18":
        base_path = "runs/Experiment2/lwf_slimres18"
        path = best_lwf_slimres18_configs[teacher_config]
    else:
        config = best_eeil_configs[teacher_config]
        base_path = "runs/Experiment2/Best_teacher_runs"
        path = f"epochs={config['epochs']}_lr={config['lr']}_mem_size={config['mem_size']}_sched_epoch={config['sched_epochs']}_ft_epoch={config['ft_epochs']}_temp={config['temp']}_optim={config['optim']}_mom={config['momentum']}"
    
    name = f"Experience{0}.pt"
    rep_data = torch.load(os.path.join(base_path, path, name), map_location=device)["rep_data"]

    return rep_data["classes_order"]

def load_teacher_checkpoints(teacher_type, teacher_config, device='cuda:0', number_of_experiences=10):

    checkpoints = []

    if teacher_type == "lwf_res32":
        base_path = "runs/Experiment2/lwf_res32"
        path = best_lwf_res32_configs[teacher_config]
    elif teacher_type == "lwf_slimres18":
        base_path = "runs/Experiment2/lwf_slimres18"
        path = best_lwf_slimres18_configs[teacher_config]
    else:
        config = best_eeil_configs[teacher_config]
        base_path = "runs/Experiment2/Best_teacher_runs"
        path = f"epochs={config['epochs']}_lr={config['lr']}_mem_size={config['mem_size']}_sched_epoch={config['sched_epochs']}_ft_epoch={config['ft_epochs']}_temp={config['temp']}_optim={config['optim']}_mom={config['momentum']}"
    
    path = os.path.join(base_path, path)

    for i in range(number_of_experiences):
        name = f"Experience{i}.pt"
        checkpoints.append(torch.load(os.path.join(path, name), map_location=device)["model_state_dict"])

    return checkpoints