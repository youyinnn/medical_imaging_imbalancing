import torch


def get_device(use_cpu=False):
    if use_cpu:
        return torch.device('cpu')
    else:
        return torch.device("cuda:0" if torch.cuda.is_available() else "mps")
