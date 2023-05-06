import torch 
import os
from models.customresnets import CustomSlimResnet18
from avalanche.models import SlimResNet18
from feature_extraction import load_dataset
from torch.utils.data import DataLoader
from avalanche.evaluation.metrics import Accuracy
from torch.nn.functional import softmax
import argparse
import json


""" Evaluate model on SplitCifar100 test given the model path"""


def models_insubpath(target_paths, model_type="res18"):
    avg = 0
    n = 0
    device = 'cuda:0'
    dataset = load_dataset(train=False)
    dataloader = DataLoader(dataset, batch_size=512)
    if model_type == "resnet18":
        model = SlimResNet18(100)
    else:
        model = CustomSlimResnet18(100)
    model.to(device)
    for path in target_paths:
        subpaths = [x[0] for x in os.walk(path)]
        print(os.path.relpath(subpaths[0], start=os.path.join(".", "runs")))
        result = {}
        for subpath in subpaths[1:]:
            try:
                model.load_state_dict(torch.load(os.path.join(subpath, "Experience9.pt"), map_location=device)['model_state_dict'])
            except FileNotFoundError:
                print("File not found")
                continue
            model.eval()
            acc = Accuracy()
            device = torch.device("cuda:0")
            for image, label in dataloader:
                image, label = image.to(device), label.to(device)
                if model_type == "resnet18":
                    acc.update(softmax(model(image), dim=1).round(), label, 0)
                else:
                    acc.update(softmax(model(image)[0], dim=1).round(), label, 0)
            result[os.path.join(path,subpath)] = acc.result()
            avg += acc.result()[0]
            n+=1
        config_path = os.path.join(".","test_outputs", os.path.relpath(subpaths[0], start=os.path.join(".", "runs", "Experiment2")))
        os.makedirs(config_path, exist_ok=True)
        print("Path created: \n", config_path)
        with open(f"{config_path}/test_results.txt", "w") as file:
            file.write(''.join([f"{value} \t: {key}\n" for key, value in result.items()]))
        print("File written")
        print("Average accuracy: ", avg/n)
    torch.cuda.empty_cache()
    return result



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", "-s", help="Student mode or not", default="slimres18")
    parser.add_argument("--paths", "-p", help="Path of the model", nargs='+')
    
    args = parser.parse_args()
    if args.paths is None:
        raise ValueError("Please provide one or more paths")
    models_insubpath(args.paths, args.student)