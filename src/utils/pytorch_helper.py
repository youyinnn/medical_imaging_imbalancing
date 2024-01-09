from torchvision.transforms.v2 import GaussianBlur
import importlib
import torch
from utils import gradient


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


class X_Aug:

    def __init__(self, model, th=0.3, blur_kernel_size=9,
                 saliency_map_fn='guided_absolute_grad', **fn_kwargs) -> None:
        self.model = model
        self.th = th
        self.blur_kernel_size = blur_kernel_size
        self.saliency_map_fn = getattr(gradient, saliency_map_fn)
        self.fn_kwargs = fn_kwargs

    def __call__(self, batch) -> torch.Any:
        exp = gradient.guided_absolute_grad(
            self.model, batch[0], batch[1],
            **self.fn_kwargs)
        n, w, h = exp.shape
        q = torch.quantile(exp.reshape(n, w * h),
                           self.th, dim=1, keepdim=True).repeat(1, w * h)
        exp = torch.where(exp > q.reshape(n, w, h), 1, 0)
        blurrer = GaussianBlur(self.blur_kernel_size)
        exp = blurrer(exp)

        n, w, h = exp.shape
        exp = exp.reshape(n, 1, w, h).repeat(1, 3, 1, 1)
        masked = batch[0] * exp
        batch[0] = masked
        return batch
