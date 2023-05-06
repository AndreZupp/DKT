import sys 
import json
from src.splitcifar100_pretrained import test
import argparse

def main(keep_class_order):
    print("Using default parameters\n ") 
    print("Teacher/Student class order shuffled: {} \n Use the --keep_order flag to change this behaviour".format(not keep_class_order)) 
    test(800, verbose=True, maintain_class_order=keep_class_order)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep_order", help="Wheter or not to use the teacher class order", action="store_true")
    print()
    args = parser.parse_args()
    main(args.keep_order)