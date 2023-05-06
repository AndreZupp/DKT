import torch
import os

def save_checkpoint(strategy, path, teacher_number, rep_data):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    torch.save({
        'experience': strategy.experience.current_experience,
        'model_state_dict': strategy.model.state_dict(),
        'optimizer_state_dict': strategy.optimizer.state_dict(),
        'loss': strategy.loss,
        'teacher_number': teacher_number,
        'rep_data' : rep_data['classes_order']
    }, f"{path}/Experience{strategy.experience.current_experience}.pt")

def dict_from_path(path, sep = "/", sep2 = "=", to_remove = None):
    parameters = path.split(sep)
    parameters = [x.split(sep2) for x in parameters]
    if to_remove is not None:
        parameters.remove([to_remove])
    parameters = dict([(x,y) for x, y in parameters])
    return parameters

def print_dict(dictionary, title=""):
    avg_length = int(sum(map(len, dictionary.keys()))/len(dictionary.keys()))
    avg_length *= 2
    head_string = '-'*avg_length + title + '-'*avg_length
    print(head_string)
    for key, value in dictionary.items():
        print(f"{key.ljust(avg_length)} {value}")
    print('-'*len(head_string))
