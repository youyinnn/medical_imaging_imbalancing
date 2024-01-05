# main.py
from src.cifar_100.cifar_dataset import CIFAR100DataModule
from src.utils.lightning_model import LightningClassifier, MyLightningCLI


def cli_main():
    MyLightningCLI(LightningClassifier, CIFAR100DataModule)
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()
