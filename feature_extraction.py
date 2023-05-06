import os 
import torch
from numpy.random import default_rng, randint
import argparse
from models.fake_model import FakeModel
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from datetime import datetime 
from models.from_avalanche import SlimResNet18
from models.customresnets import CustomSlimResnet18

""" Extract the output of a model's feature extractor """


def list_to_chunks(l, n=10):
    result = []
    for i in range(0, len(l), n):
        result.append(l[i:(i+n)])
    return result


def load_dataset(train=False):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
            ])
    return datasets.CIFAR100('./data/', train=train, download=True, transform=train_transform)



def print_model(model):
    children_counter = 0
    for n,c in model.named_children():
        print("Children Counter: ",children_counter," Layer Name: ",n)
        children_counter+=1

def get_class_per_expereince(student, model_path): 
    if model_path is None:
        if not student:
            path = os.path.join(".", "runs", "Experiment2", "lwf_slimres18", "epochs=50_lr=0.01_temp=2_optim=adam_lr_sched=0_alpha=1_early_stop=True")
        else:
            path = os.path.join(".", "runs", "Experiment2", "shuffled_for_cka", "teacher=lwf_slimres18_student=slimres18_epochs=900_lr=0.01_kd_rels=0.01_temp=2_cm_prob=0_lr_sched=0")
    else:
        path = model_path
    checkpoint = torch.load(os.path.join(path, "Experience0.pt"))
    class_order = checkpoint["rep_data"]
    if isinstance(class_order, dict):
        class_order = class_order["classes_order"]
    res = list_to_chunks(class_order)
    print("Classes per experience")
    for i, elem in enumerate(res):
        print(f"Experience{i}\n {elem}\n")


def class_evaluation(class_ids, batch_size, model_path, save_path, student=False):

    """
    This function samples two random classes and evaluates their representation througout the training process 
    """
    class_ids = list(map(int, class_ids))
    
    if model_path is None:
        if not student:
            path = os.path.join(".", "runs", "Experiment2", "lwf_slimres18", "epochs=50_lr=0.01_temp=2_optim=adam_lr_sched=0_alpha=1_early_stop=True")
        else:
            path = os.path.join(".", "runs", "Experiment2", "shuffled_for_cka", "teacher=lwf_slimres18_student=slimres18_epochs=900_lr=0.01_kd_rels=0.01_temp=2_cm_prob=0_lr_sched=0")
    else:
        path = model_path
    if len(class_ids) == 1:
        class_ids = [randint(100) for _ in range(class_ids[0])]
    
    print(f"Targetting the classes: {class_ids}")

    dataset = load_dataset()

    if not student:
        model = SlimResNet18(100)
    else:
        model = CustomSlimResnet18(100)
    subset = Subset(dataset, [i for i, (x,y) in enumerate(dataset) if y in class_ids])
    loader = DataLoader(subset, batch_size=batch_size)
    result = {}
    for i in range(10):
        checkpoints = torch.load(os.path.join(path, f"Experience{i}.pt"), map_location='cuda:0')
        model.load_state_dict(checkpoints['model_state_dict'])
        if not student:
            fake_model = FakeModel(model, "layer4")
        else:
            fake_model = FakeModel(model.slim_res18, "layer4")
        for image, labels in loader:
            result[f"Experience{i}"] = fake_model(image)
            target = labels
            break
            

    now = datetime.now().strftime("%H:%M:%S")
    if save_path is None:
        filename = f"model_outputs/{now}.pt"
        target_filename = f"target_outputs/{now}.pt"
    else:
        filename = os.path.join(".","tsne_data","class_evaluation", "model_outputs",save_path)
        target_filename = os.path.join(".", "tsne_data","class_evaluation", "targets", save_path)

    if not os.path.exists(os.path.join(".", "tsne_data","class_evaluation", "targets")):
        os.makedirs(os.path.join(".","tsne_data","class_evaluation", "model_outputs"), exist_ok=True)
        os.makedirs(os.path.join(".", "tsne_data","class_evaluation", "targets"), exist_ok=True)
    torch.save(target, target_filename)
    torch.save(result, filename)
    print("The model output has been saved in {}".format(filename))




def kd_effect(class_ids, batch_size, model_path, save_path, student=False):
    """
    This function identifies randomly "class_ids" classes that are only present in the last experience and compares their 
    representation througout the training process, up to the last experience. 
    """
    class_ids = list(map(int, class_ids))

    if len(class_ids) > 1:
        raise ValueError("Too many arguments for the classes parameter.\n With mode kd you only have to specify the 1 <= n <= 10 classes to sample from the last experience")

    if model_path is not None:
        path = model_path
    else:
        path = os.path.join(".", "runs", "Experiment2", "lwf_slimres18", "epochs=50_lr=0.01_temp=2_optim=adam_lr_sched=0_alpha=1_early_stop=True")
    
    checkpoints = [torch.load(os.path.join(path, f"Experience{i}.pt"), map_location='cuda:0') for i in range(10)]
    last_exp_classes = list_to_chunks(checkpoints[0]['rep_data'])
    last_exp_classes = last_exp_classes[-1]
    rng = default_rng()
    indices = rng.choice(10, size=class_ids[0], replace=False).tolist()
    class_ids = [last_exp_classes[indx] for indx in indices]
    
    dataset = load_dataset()

    if not student:
        model = SlimResNet18(100)
    else:
        model = CustomSlimResnet18(100)
 
    subset = Subset(dataset, [i for i, (x,y) in enumerate(dataset) if y in class_ids])
    loader = DataLoader(subset, batch_size=batch_size)
    result = {}
    
    print(f"Targetting the classes: {class_ids}")

    for i in range(10):
        model.load_state_dict(checkpoints[i]['model_state_dict'])
        if student:
            fake_model = FakeModel(model.slim_res18, "layer4")
        else:
            fake_model = FakeModel(model, "layer4")

        for image, labels in loader:
            result[f"Experience{i}"] = fake_model(image)
            target = labels
            break
            
    now = datetime.now().strftime("%H:%M:%S")
    if save_path is None:
        filename = f"model_outputs/{now}.pt"
        target_filename = f"target_outputs/{now}.pt"
    else:
        filename = os.path.join(".","tsne_data","kd_effect","model_outputs",save_path)
        target_filename = os.path.join(".", "tsne_data","kd_effect", "targets", save_path)

    if not os.path.exists(os.path.join(".", "tsne_data","kd_effect", "targets")):
        os.makedirs(os.path.join(".","tsne_data","kd_effect","model_outputs"), exist_ok=True)
        os.makedirs(os.path.join(".", "tsne_data","kd_effect", "targets"), exist_ok=True)
    torch.save(target, target_filename)
    torch.save(result, filename)
    print("The model output has been saved in {}".format(filename))

def random_extraction(samples, model_path, student=False, whole_dataset=False):

    if model_path is not None:
        path = model_path
    else:
        path = os.path.join(".", "runs", "Experiment2", "lwf_slimres18", "epochs=50_lr=0.01_temp=2_optim=adam_lr_sched=0_alpha=1_early_stop=True")

    

    dataset = load_dataset()
    data = DataLoader(dataset, batch_size=samples, shuffle=True)
    
    result = {}
    if not student:
        model = SlimResNet18(100)
    else:
        model = CustomSlimResnet18(100)

    
    for i in range(10):
        checkpoints = torch.load(os.path.join(path, f"Experience{i}.pt"), map_location='cuda:0')
        model.load_state_dict(checkpoints['model_state_dict'])
        fake_model = FakeModel(model, "layer4")
        for image, labels in data:
            result[f"Experience{i}"] = fake_model(image)
            target = labels
            if not whole_dataset:
                break
    
    now = datetime.now().strftime("%H:%M:%S")
    filename = f"model_outputs/{now}.pt"
    target_filename = f"target_outputs/{now}.pt"
    torch.save(result, filename)
    torch.save(target, target_filename)
    
    print("The model output has been saved in {}".format(filename))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help="Select between class or random")
    parser.add_argument("-c", "--classes", help="classes to sample", nargs='+')
    parser.add_argument("-n", "--n_samples", help="number_of_samples", type=int)
    parser.add_argument("-s", "--save", help="Save path")
    parser.add_argument("-p", "--path", help="Path of the model")
    parser.add_argument('--student', action='store_true')
    parser.add_argument('--no-student', dest='student', action='store_false')
    parser.set_defaults(student=False)
    args = parser.parse_args()



    if args.mode == "class":
        class_evaluation(args.classes, int(args.n_samples), args.path, args.save, student=args.student)
    elif args.mode == "kd":
        kd_effect(args.classes, int(args.n_samples), args.path, args.save, student=args.student)
    elif args.mode == "class_list":
        get_class_per_expereince(args.student, args.path)
    else:
        random_extraction(int(args.n_samples))

