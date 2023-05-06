import torch_cka
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from feature_extraction import load_dataset
from models.resnet import resnet34
from models.customresnets import CustomSlimResnet18
from avalanche.models import SlimResNet18
from utils.loaders import load_teacher_checkpoints
import argparse

""" CKA given model """


def get_layer_list(model, student=True):
    ll = []
    for name, layer in model.named_modules():
        ll.append(name)
    return ll 

def cka_experiment1(model2_path, train, filename):
  
    model1_path = os.path.join(".", "runs", "Experiment1", "resnet34", "resnet34.pth")

    if model2_path is None:
        #/home/a.zuppolini/CL/runs/Experiment1/cutmix_trials/epoch=700/cutmix_prob=0.8/kd_rel=20/loss=mse
        model2_path = os.path.join(".", "runs", "Experiment1", "cutmix_trials",
                                    "epoch=700", "cutmix_prob=0.8", "kd_rel=20", "loss=mse")
    else:
        model2_path = model2_path

    dataset = load_dataset(train = train)
    dataloader = DataLoader(dataset, 512)

    model1 = resnet34()
    model1.load_state_dict(torch.load(model1_path, map_location="cuda:0"))
    model2 = CustomSlimResnet18(100)
    model2_cp = [torch.load(os.path.join(model2_path, f"Experience{i}.pth")) for i in range(10)]
    model1_ll = get_layer_list(model1)
    model2_ll = get_layer_list(model2.slim_res18)
    result = {}
    
    for i in range(10):
        model2.load_state_dict(model2_cp[i]['model_state_dict'])
        cka = torch_cka.CKA(model1=model1, model2=model2.slim_res18, model1_name="teacher", model2_name="student",
                            model1_layers=model1_ll[:-2], model2_layers=model2_ll[:-1], device='cuda:0')
        cka.compare(dataloader)
        result[f"experience{i}"] = cka.export()['CKA']

    os.makedirs("./cka/Whole_comparison/Experiment1", exist_ok=True)
    torch.save(result, f"cka/Whole_comparison/Experiment1/{filename}")

    print("Cka results saved in ", f"cka/comparison/{filename}")



def whole_comparison(model1_path, model1_name, train, filename):

    if model1_path is None:
        print("Using lwf_slimres18 as model 1")
        model1_path = os.path.join(".", "runs", "Experiment2", "lwf_slimres18",
                                    "epochs=50_lr=0.01_temp=2_optim=adam_lr_sched=30_alpha=1_early_stop=True")
    
    model2_path = os.path.join(".", "runs", "Experiment2", "shuffled_for_cka",
                                "teacher=lwf_slimres18_student=slimres18_epochs=900_lr=0.01_kd_rels=0.01_temp=2_cm_prob=0_lr_sched=0")
    
    model1 = SlimResNet18(100)
    model2 = CustomSlimResnet18(100)

    model1_cp = [torch.load(os.path.join(model1_path, f"Experience{x}.pt"), map_location="cuda:0") for x in range(10)]
    model2_cp = [torch.load(os.path.join(model2_path, f"Experience{x}.pt"), map_location="cuda:0") for x in range(10)]

    model1_ll = get_layer_list(model1)
    model2_ll = get_layer_list(model2.slim_res18)

    dataset = load_dataset(train = train)
    dataloader = DataLoader(dataset, 512)

    result = {}
    for i in range(10):

        model1.load_state_dict(model1_cp[i]['model_state_dict'])
        model2.load_state_dict(model2_cp[i]['model_state_dict'])

        cka = torch_cka.CKA(model1=model1, model2=model2.slim_res18, model1_layers=model1_ll[:-1], model2_layers=model2_ll[:-1],
                            model1_name=model1_name, model2_name="Student", device="cuda:0")

        cka.compare(dataloader)
        result[f"experience{i}"] = cka.export()['CKA']


    os.makedirs("./cka/Whole_comparison/", exist_ok=True)
    torch.save(result, f"cka/Whole_comparison/Experiment2/{filename}")

    print("Cka results saved in ", f"cka/comparison/Experiment2/{filename}")


def compute_cka(model1_path, model1_name, train, filename, experience, student=False):

    if experience is None:
        experience = 9
    else:
        experience = int(experience)

    if model1_path is None:
        print("Using lwf_slimres18 as model 1")
        model1_path = os.path.join(".", "runs", "Experiment2", "lwf_slimres18",
                                    "epochs=50_lr=0.01_temp=2_optim=adam_lr_sched=30_alpha=1_early_stop=True")
    
    model2_path = os.path.join(".", "runs", "Experiment2", "shuffled_for_cka",
                                "teacher=lwf_slimres18_student=slimres18_epochs=900_lr=0.01_kd_rels=0.01_temp=2_cm_prob=0_lr_sched=0")
    
    if student is True:
        model1= CustomSlimResnet18(100)
    else:    
        model1 = SlimResNet18(100)
    model2 = CustomSlimResnet18(100)

    model1_cp = torch.load(os.path.join(model1_path, f"Experience{experience}.pt"), map_location="cuda:0")
    model2_cp = torch.load(os.path.join(model2_path, f"Experience{experience}.pt"), map_location="cuda:0")

    model1.load_state_dict(model1_cp['model_state_dict'])
    model2.load_state_dict(model2_cp['model_state_dict'])
    if student is True:
        model1_ll = get_layer_list(model1.slim_res18)
    else:
        model1_ll = get_layer_list(model1)
    model2_ll = get_layer_list(model2.slim_res18)

    dataset = load_dataset(train = train)
    dataloader = DataLoader(dataset, 512)
    if student is True:
        cka = torch_cka.CKA(model1=model1.slim_res18, model2=model2.slim_res18, model1_name=model1_name, model2_name="Student", device="cuda:0")
    else:
        cka = torch_cka.CKA(model1=model1, model2=model2.slim_res18, model1_name=model1_name, model2_name="Student",
                            model1_layers=model1_ll[:-1], model2_layers=model2_ll[:-1], device="cuda:0")
    cka.compare(dataloader)
    result = cka.export()
    os.makedirs("./cka/comparison/", exist_ok=True)
    torch.save(result, f"cka/comparison/{filename}")
    print("Cka results saved in ", f"cka/comparison/{filename}")

    return cka

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--whole", action="store_true")
    parser.add_argument("--no-whole", action="store_false", dest="whole")
    parser.add_argument("-p", "--path", help="Model1 path")
    parser.add_argument("-n", "--name", help="Model1 name")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--no-train", action="store_false", dest="train")
    parser.add_argument("-s", "--save", help="path where the file will be saved")
    parser.add_argument("-e", "--experience", type=int)
    parser.add_argument("--experiment1", action="store_true")
    parser.add_argument("--no-experiment1", action="store_false", dest="experiment1")
    parser.add_argument("--seed", help="seed for reproducibility")

    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(int(args.seed))

    if args.experiment1 is True:
        cka_experiment1(args.path, args.train, args.save)
    elif args.whole == True:
        whole_comparison(args.path, args.name, args.train, args.save)
    else:
        compute_cka(args.path, args.name, args.train, args.save, args.experience)
