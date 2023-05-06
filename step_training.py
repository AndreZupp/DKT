from src.step_training import test
import argparse

"""Experiment 3: Two models trained on different streams using Distributed Knowledge Transfer (DKT)"""


def main():
    print("Using default arguments")
    test(training_epochs=800, verbose=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", "-e", help="Number of epochs", type=int)
    parser.add_argument("--kd", nargs='?', default=0.01, type=float)
    parser.add_argument("--lrsched", nargs='?', default=0, type=int)
    parser.add_argument("--gpu", nargs='?', default=0, type=int)
    parser.add_argument("--early-stop", nargs='?', default=-1, type=int)
    parser.add_argument("--auto", default=True, help="This flag will use the configuration present in the main", action="store_true")
    parser.add_argument("--no-auto", action="store_false", dest="auto")
    args = parser.parse_args()

    if args.auto is True:
        main()
    elif args.epochs is None or not isinstance(args.epochs, int) or args.epochs == 0:
        raise ValueError("Please provide a positivie integer as number of epochs")
    else:
        test(args.epochs, sched_epochs=args.lrsched, kd_rel=args.kd, device_num=args.gpu, logdir="", early_stopping=args.early_stop, eval_on_test=False)
