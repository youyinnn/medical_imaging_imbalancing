
import os
import numpy as np
import torch
from PIL import Image
import random

# ANCHOR: Data


def fix_seed(s=0):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    random.seed(s)
    np.random.seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normm_i_t(tensors):
    inputs_shape, _, w, h = tensors.shape
    mean_i_t = torch.stack([torch.full((inputs_shape,  w, h), 0.485), torch.full(
        (inputs_shape,  w, h), 0.456), torch.full((inputs_shape,  w, h), 0.406)], dim=1).to(tensors.device)
    std_i_t = torch.stack([torch.full((inputs_shape, w, h), 0.229), torch.full(
        (inputs_shape, w, h), 0.224), torch.full((inputs_shape, w, h), 0.225)], dim=1).to(tensors.device)
    return (tensors - mean_i_t) / std_i_t


def denormm_i_t(tensors):
    inputs_shape, _, w, h = tensors.shape
    mean_i_t = torch.stack([torch.full((inputs_shape,  w, h), 0.485), torch.full(
        (inputs_shape,  w, h), 0.456), torch.full((inputs_shape,  w, h), 0.406)], dim=1).to(tensors.device)
    std_i_t = torch.stack([torch.full((inputs_shape, w, h), 0.229), torch.full(
        (inputs_shape, w, h), 0.224), torch.full((inputs_shape, w, h), 0.225)], dim=1).to(tensors.device)
    return tensors * std_i_t + mean_i_t


def normm(np_arr):
    return (np_arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])


def normm_i(np_arr):
    inputs_shape = np_arr.shape[0]
    mean_i = np.stack([np.full((inputs_shape,  256, 256), 0.485), np.full(
        (inputs_shape,  256, 256), 0.456), torch.full((inputs_shape,  256, 256), 0.406)], axis=1)
    std_i = np.stack([np.full((inputs_shape, 256, 256), 0.229), np.full(
        (inputs_shape, 256, 256), 0.224), np.full((inputs_shape, 256, 256), 0.225)], axis=1)

    return (np_arr - mean_i) / std_i


def denormm(np_arr):
    return np_arr * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])


def denormm_i(np_arr):
    inputs_shape = np_arr.shape[0]
    mean_i = np.stack([
        np.full((inputs_shape,  256, 256), 0.485),
        np.full((inputs_shape,  256, 256), 0.456),
        torch.full((inputs_shape,  256, 256), 0.406)],
        axis=1)
    std_i = np.stack([np.full((inputs_shape, 256, 256), 0.229), np.full(
        (inputs_shape, 256, 256), 0.224), np.full((inputs_shape, 256, 256), 0.225)], axis=1)

    return np_arr * std_i + mean_i


def entropy(gray):
    hist, _ = np.histogram(gray, bins=256)
    hist = hist / float(gray.size)
    hist = hist[hist != 0]
    entropy_value = -np.sum(hist * np.log2(hist))
    return entropy_value


def entropy_rgb(rgb):
    return np.array([entropy(rgb[i]) for i in range(3)]).mean()


def entropy_change(images, mixed):
    imgaes_np = images.to('cpu').detach().numpy()
    mixed_np = mixed.to('cpu').detach().numpy()

    images_entropy = np.array([entropy_rgb(img) for img in imgaes_np])
    mixed_entropy = np.array([entropy_rgb(img) for img in mixed_np])
    # images_entropy = np.array([entropy(img) for img in imgaes_np])
    # mixed_entropy = np.array([entropy(img) for img in mixed_np])

    return images_entropy - mixed_entropy


mean_t = torch.tensor([0.485, 0.456, 0.406])
std_t = torch.tensor([0.229, 0.224, 0.225])


def build_input(inputs, device, norm=False):
    if norm:
        input = (torch.tensor(np.array([input.astype(
            np.float32) for input in inputs]), device=device) - mean_t.to(device)) / mean_t.to(device)
    else:
        input = torch.tensor(
            np.array([input.astype(np.float32) for input in inputs]), device=device)
    return input.transpose(2, 3).transpose(1, 2)


def read_array(instance_path, read_file_name):
    npz = os.path.exists(os.path.join(instance_path, 'maps.npz'))
    ext = '.npz' if npz else '.npy'

    with open(os.path.join(instance_path, f'{read_file_name}{ext}'), 'rb') as f:
        array = np.load(f, allow_pickle=True)[
            'arr_0'] if npz else np.load(f, allow_pickle=True)

    return array


def norm_noise(sigma, n, fold=True, channel=3):
    mu = 1  # mean
    # Generate the samples from the normal distribution
    if channel == 3:
        samples = np.random.normal(
            mu, sigma, size=(n, 3, 256, 256)).astype(np.float32)
    else:
        samples = np.random.normal(
            mu, sigma, size=(n, 256, 256)).astype(np.float32)
        samples = np.stack([samples for i in range(3)], axis=1)

    samples = np.where(samples < 0, np.abs(samples), samples)
    if not fold:
        samples = np.where(samples > 2, 2 - (samples - 2), samples)
    if fold:
        samples = np.where(samples < 0, np.abs(samples), samples)
        samples = np.where(samples > 1, np.abs(2 - samples), samples)
    return samples


def min_max_norm(u):
    if u.sum() == 0 or len(u) < 2:
        return u
    u -= u.min()
    u /= u.max()
    return u


def min_max_norm_matrix(u, axis=None):
    if type(u) is torch.Tensor:
        umin = u.min(dim=-1, keepdim=True).values.min(dim=-
                                                      2, keepdim=True).values
        u -= umin
        umax = u.max(dim=-1, keepdim=True).values.max(dim=-
                                                      2, keepdim=True).values
        u /= umax
        if torch.isnan(u).all():
            u = torch.ones_like(u)
    else:
        # narrays
        u -= u.min(axis=axis, keepdims=True)
        u /= u.max(axis=axis, keepdims=True)
        if np.isnan(u).all():
            u = np.ones_like(u)
    return torch.nan_to_num(u)


def mask(ori, sal):
    # sal = (sal + 0.1)
    if ori.shape[0] == 3:
        ori = np.transpose(ori, (1, 2, 0))
    sal = np.where(sal > 1, 1, sal)
    g = ori * np.array(Image.fromarray(sal * 255).convert('RGB'))
    return g / 255
    # return ori


def get_images_targets(dataset, device, ran_idx):
    if dataset is None:
        return None, None
    images = torch.stack([dataset[i][0].to(device) for i in ran_idx])
    if isinstance(dataset[0][1], dict):
        targets = [dataset[i][1] for i in ran_idx]
    else:
        targets = torch.tensor([dataset[i][1]
                                for i in ran_idx], device=device)

    return images, targets
