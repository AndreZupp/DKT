from src.naive_finetuning import test
import sys


if __name__ == "__main__":
    for i in range(5):
        test(150, architecture="slimres18", logdir="custom_avg_variance", filename_suffix=f"run{i}", device_num=0)

    