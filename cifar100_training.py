from src.cifar100 import test
import sys

#Config 2 =200epochs cutmix_prob=0.5/kd_rel=10/temperature=1/loss=mse Accuracy on 3rd exp: 0.81 (0.83 on Student head)

if __name__ == "__main__":
    print("Using default parameters")  
    test(300, verbose=True)


    

