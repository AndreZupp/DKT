from src.lwf import test
import sys 
import json
def main():

    grid_search = False
    device_num = sys.argv[1]

    if grid_search:
        epochs = [50, 70, 90, 120]
        lr_scheds = [0, 30, 40]
        alphas = [1, 1.5, 2]
        early_stoppings = [True, False]
        optimizer_names = ['adam']

        for early_stopping in early_stoppings:
            for lr_sched in lr_scheds:
                for alpha in alphas:
                    for epoch in epochs:
                        for optimizer_name in optimizer_names:
                            test(training_epochs=epoch, optimizer_name=optimizer_name, teacher_fe="slimres18", alpha=alpha, lr=0.01, device_num=device_num, logdir="lwf_slimres18", temperature=2, mem_size=2000, early_stopping=early_stopping, lr_scheduler=lr_sched)
    else:
        metrics = []
        for i in range(5, 10):
            metrics.append(test(training_epochs=50, optimizer_name="adam", lr=0.01, temperature=2, lr_scheduler=30, early_stopping=True, alpha=1,  device_num=device_num, filename_suffix=f"run_{i}", teacher_fe="slimres18", logdir="lwf_slimres18_best_runs", mem_size=2000))
        with open("./logs/lwf_slimres18_best_runs.txt", "w") as file:
            file.write(json.dumps(metrics))





if __name__ == "__main__":
    main()