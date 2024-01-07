import importlib
import torch


def get_device(use_cpu=False):
    return torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


# module_name, class_name = "module.submodule.MyClass".rsplit(".", 1)
# MyClass = getattr(importlib.import_module(module_name), class_name)

def init_transform(class_path: str, init_args):
    module_name, class_name = class_path.rsplit(".", 1)

    tran = getattr(importlib.import_module(module_name), class_name)
    if type(init_args) is dict:
        return tran(**init_args)
    if type(init_args) is list:
        args = []
        for a in init_args:
            if type(a) is dict and a.get('class_path') is not None:
                args.append(init_transform(a['class_path'], a['init_args']))
            else:
                args.append(a)
        return tran(args)
