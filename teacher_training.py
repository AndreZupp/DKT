import torch.cuda
from src.eeil import test
import sys
from datetime import datetime

"""EEIL (End-to-End Incremental Learning) grid search"""

grid_search = True
def insert_in_top5(top_dict, accuracy_value, model_config):
    if len(top_dict) < 5:
        top5[accuracy_value] = model_config
    else:
        accuracy_list = list(top_dict.keys())
        accuracy_list.sort()
        min_acc = accuracy_list[0]
        if accuracy_value > min_acc:
            top_dict.pop(min_acc)
            top_dict[accuracy_value] = model_config


if __name__ == "__main__":

    if grid_search:
        filename = "top_accuracy " + datetime.now().strftime("%d-%m-%Y %H:%M") + ".txt"
        epochs = [70, 90, 100, 140]
        mem_sizes = [2000, 3000]
        strategy = "custom"
        lrs = [0.01]
        optimizers = ['sgd']
        momentums = [0.9, 0.5]
        scheduler_epochs = [30]
        finetuning_epochs = [30]
        temperatures = [2, 3]
        alphas = [1, 1.5]
        top5 = dict()
        device_num = sys.argv[1]

        logdir = "Teacher training"

        for epoch in epochs:
            for lr in lrs:
                for mem_size in mem_sizes:
                    for scheduler_epoch in scheduler_epochs:
                        for finetuning_epoch in finetuning_epochs:
                            for temperature in temperatures:
                                for optimizer in optimizers:
                                    for alpha in alphas:
                                        if optimizer == "sgd":
                                            for momentum in momentums:
                                                config = f"epochs={epoch}/lr={lr}/mem_size={mem_size}/sched_epoch={scheduler_epoch}/ft_epoch={finetuning_epoch}/temp={temperature}/optim={optimizer}/mom={momentum}/alpha={alpha}"
                                                metric_values = test(training_epochs=epoch, mem_size=mem_size, lr=lr, device_num=device_num,
                                                                    scheduler_epochs=scheduler_epoch, finetuning_epochs=finetuning_epoch, temperature=temperature,
                                                                    optimizer_name=optimizer, momentum=momentum, filename_suffix=config.replace('/', '_'), logdir=logdir, strategy_type=strategy, alpha=alpha)
                                                accuracy = list(metric_values.values())[1]
                                                insert_in_top5(top5, accuracy, config)
                                        else:
                                                config = f"epochs={epoch}/lr={lr}/mem_size={mem_size}/sched_epoch={scheduler_epoch}/ft_epoch={finetuning_epoch}/temp={temperature}/optim={optimizer}/alpha={alpha}"
                                                metric_values = test(training_epochs=epoch, mem_size=mem_size, lr=lr, device_num=device_num,
                                                                    scheduler_epochs=scheduler_epoch, finetuning_epochs=finetuning_epoch, temperature=temperature,
                                                                    optimizer_name=optimizer, filename_suffix=config.replace('/', '_'), logdir=logdir, strategy_type=strategy, alpha=alpha)
                                                accuracy = list(metric_values.values())[1]
                                                insert_in_top5(top5, accuracy, config)

        with open(filename, "w") as f:
            for key, value in top5.items():
                f.write('%s:%s\n' % (key, value))
    else:
        test(70, mem_size=2000, lr=0.01)

    torch.cuda.empty_cache()
