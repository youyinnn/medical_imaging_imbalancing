import json
import os
import sys
import skorch

import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler, EpochScoring, PrintLog
from sklearn.metrics import top_k_accuracy_score
from utils.pytorch_helper import get_device
from utils.skorch_callbacks import *


def net_def(
    net_class: nn.Module, net_name,
    classes, classifier_kwargs: dict,
    cut_mixed=False
):
    net_name = net_name + '_'

    classifier_kwargs.setdefault('lr', 0.001)
    classifier_kwargs.setdefault('criterion', nn.CrossEntropyLoss)
    classifier_kwargs.setdefault('batch_size', 256)
    classifier_kwargs.setdefault('optimizer', optim.SGD)
    # classifier_kwargs.setdefault('optimizer', optim.Adam)
    classifier_kwargs.setdefault('optimizer__momentum', 0.9)
    classifier_kwargs.setdefault('iterator_train__shuffle', True)

    classifier_kwargs.setdefault('iterator_train__num_workers', 1)
    classifier_kwargs.setdefault('iterator_train__pin_memory', True)
    classifier_kwargs.setdefault('iterator_train__persistent_workers', True)
    classifier_kwargs.setdefault('iterator_train__prefetch_factor', 32)

    classifier_kwargs.setdefault('iterator_valid__num_workers', 1)
    classifier_kwargs.setdefault('iterator_valid__pin_memory', True)
    classifier_kwargs.setdefault('iterator_valid__persistent_workers', True)
    classifier_kwargs.setdefault('iterator_valid__prefetch_factor', 32)

    classifier_kwargs.setdefault('device', get_device())

    cp_fn_prefix = os.path.join('weights', net_name, net_name)

    def top_k(net: NeuralNetClassifier, ds, y=None, k=1):
        # assume ds yields (X, y), e.g. torchvision.datasets.MNIST
        y_true = [y for _, y in ds]
        y_pred = net.predict_proba(ds)
        return top_k_accuracy_score(y_true, y_pred, k=k,
                                    labels=[i for i in range(classifier_kwargs['module__output_features'])])

    default_callbacks = [
        EpochScoring(scoring='f1_macro', name='valid_f1',
                     lower_is_better=False),
        # EpochScoring(scoring=lambda net, ds, y: top_k(net, ds, y=y, k=1), name='valid_top_1',
        #              lower_is_better=False),
        # EpochScoring(scoring=lambda net, ds, y: top_k(net, ds, y=y, k=3), name='valid_top_3',
        #              lower_is_better=False),
        # EpochScoring(scoring=lambda net, ds, y: top_k(net, ds, y=y, k=5), name='valid_top_5',
        #              lower_is_better=False),
        # LRScheduler(policy='StepLR', step_size=7, gamma=0.1),
        MyCheckpoint(monitor='valid_f1_best', load_best=True,
                     fn_prefix=cp_fn_prefix)
    ]
    if classifier_kwargs.get('callbacks') is not None:
        default_callbacks.extend(classifier_kwargs['callbacks'])
    classifier_kwargs['callbacks'] = default_callbacks

    if classifier_kwargs.get('train_split') is not None:
        classifier_kwargs['train_split'] = predefined_split(
            classifier_kwargs.get('train_split'))

    if cut_mixed:
        classifier = CutMixedNeuralNetClassifier
    else:
        classifier = NeuralNetClassifier
    net = classifier(
        net_class,
        classes=classes,
        warm_start=True,              # continue the last fit
        **classifier_kwargs
    )

    setattr(net, 'net_name', net_name)
    setattr(net, 'cp_fn_prefix', cp_fn_prefix)
    return net


def net_fit(net: skorch.NeuralNet, x, y, max_epochs, new_lr=None, load_best=False):
    weight_path_pre = os.path.join('weights', net.net_name)
    if not os.path.exists(weight_path_pre):
        os.makedirs(weight_path_pre, exist_ok=True)

    best_prefix = "" if load_best else "last_"

    params_pt_path = os.path.join(
        weight_path_pre, f"{net.net_name}{best_prefix}params.pt")
    history_json_path = os.path.join(
        weight_path_pre, f"{net.net_name}history.json")
    optimizer_pt_path = os.path.join(
        weight_path_pre, f"{net.net_name}{best_prefix}optimizer.pt")
    criterion_pt_path = os.path.join(
        weight_path_pre, f"{net.net_name}{best_prefix}criterion.pt")
    if os.path.exists(history_json_path):
        with open(history_json_path, 'rb') as f:
            h = json.load(f)
        if net.history != h:
            print(f"Load saved params: {net.net_name}params.pt")
            net.initialize()
            pms = {}
            pms['f_params'] = params_pt_path
            pms['f_history'] = history_json_path
            if os.path.exists(optimizer_pt_path):
                pms['f_optimizer'] = optimizer_pt_path

            if os.path.exists(criterion_pt_path):
                pms['f_criterion'] = criterion_pt_path

            net.load_params(**pms)
        else:
            print(f"Current params are the latest, continue the training.")

        print('Histories:')
        p = PrintLog()
        p.initialize()
        data = net.history[0]
        verbose = net.verbose
        tabulated = p.table(data)

        current_epochs = len(net.history)

        header, lines = tabulated.split('\n', 2)[:2]
        p._sink(header, verbose)
        p._sink(lines, verbose)

        for h in net.history:
            tabulated = p.table(h)
            header, lines = tabulated.split('\n', 2)[:2]
            p._sink(tabulated.split('\n')[2], verbose)
            if p.sink is print:
                sys.stdout.flush()
        epochs = max_epochs - current_epochs
        if new_lr is not None:
            for param_group in net.optimizer_.param_groups:
                param_group['lr'] = new_lr
        if epochs > 0:
            print('\n\n', 'Continue training:')
            net.partial_fit(X=x, y=y, epochs=epochs)
    else:
        epochs = max_epochs
        if epochs > 0:
            if new_lr is not None:
                for param_group in net.optimizer_.param_groups:
                    param_group['lr'] = new_lr
            print('\n\n', 'New training:')
            net.fit(X=x, y=y, epochs=epochs)

    # print(net.history)
    return net
