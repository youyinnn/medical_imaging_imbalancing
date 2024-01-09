# main.py
# autopep8: off
import os
import sys
sys.path.insert(0, os.path.abspath('src'))
from utils.lightning_model import LightningClassifier, MyLightningCLI
from cifar_100.cifar_dataset import CIFAR100DataModule
# autopep8: on


def cli_main():
    MyLightningCLI(LightningClassifier, CIFAR100DataModule)
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()
