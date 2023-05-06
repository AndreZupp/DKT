from src.eeil import test
import sys
from utils.best_teacher_configs import best_configs

# End to End incremental Learning test


if __name__ == "__main__":
    best_config_run = False
    if best_config_run:
        for key in best_configs.keys():
            config = best_configs[key]
            test(training_epochs=config["epochs"], mem_size=config["mem_size"], lr=config["lr"], optimizer_name=config["optim"], scheduler_epochs=config["sched_epochs"],
                    finetuning=config["ft_epochs"], momentum=config["momentum"], temperature=config["temp"], device_num=0, logdir="Best_teacher_runs")
    else:
        test(training_epochs=70, verbose=True, mem_size=3000, lr=0.1, device_num=sys.argv[1], strategy_type="custom", optimizer_name="sgd",
                scheduler_epochs=30, finetuning_epochs=30, earlystopping=True, logdir="Troubleshooting", momentum=0.9, temperature=3, train=True)
