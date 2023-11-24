import json
import os
import sys
import skorch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler, EpochScoring, PrintLog

from utils.pytorch_helper import get_device

import pickle
import warnings
from skorch.callbacks import Callback
from skorch.exceptions import SkorchException
from skorch.utils import _check_f_arguments
from skorch.utils import noop
from skorch.utils import open_file_like


class MyCheckpoint(Callback):

    def __init__(
            self,
            monitor='valid_loss_best',
            f_params='params.pt',
            f_optimizer='optimizer.pt',
            f_criterion='criterion.pt',
            f_history='history.json',
            f_pickle=None,
            fn_prefix='',
            dirname='',
            event_name='event_cp',
            sink=noop,
            load_best=False,
            use_safetensors=False,
            **kwargs
    ):
        self.monitor = monitor
        self.f_params = f_params
        self.f_optimizer = f_optimizer
        self.f_criterion = f_criterion
        self.f_history = f_history
        self.f_pickle = f_pickle
        self.fn_prefix = fn_prefix
        self.dirname = dirname
        self.event_name = event_name
        self.sink = sink
        self.load_best = load_best
        self.use_safetensors = use_safetensors
        self._check_kwargs(kwargs)
        vars(self).update(**kwargs)
        self._validate_filenames()

    def _check_kwargs(self, kwargs):
        for key in kwargs:
            if not key.startswith('f_'):
                raise TypeError(
                    "{cls_name} got an unexpected argument '{key}', did you mean "
                    "'f_{key}'?".format(cls_name=self.__class__.__name__, key=key))
        if self.use_safetensors and self.f_optimizer is not None:
            raise ValueError(
                "Cannot save optimizer state when using safetensors, "
                "please set f_optimizer=None or don't use safetensors.")

    def initialize(self):
        self._validate_filenames()
        if self.dirname and not os.path.exists(self.dirname):
            os.makedirs(self.dirname, exist_ok=True)
        return self

    def on_train_end(self, net, **kwargs):
        if not self.load_best or self.monitor is None:
            return
        self._sink("Loading best checkpoint after training.", net.verbose)
        net.load_params(checkpoint=self, use_safetensors=self.use_safetensors)

    def on_epoch_end(self, net, **kwargs):
        if "{}_best".format(self.monitor) in net.history[-1]:
            warnings.warn(
                "Checkpoint monitor parameter is set to '{0}' and the history "
                "contains '{0}_best'. Perhaps you meant to set the parameter "
                "to '{0}_best'".format(self.monitor), UserWarning)

        if self.monitor is None:
            do_checkpoint = True
        elif callable(self.monitor):
            do_checkpoint = self.monitor(net)
        else:
            try:
                do_checkpoint = net.history[-1, self.monitor]
            except KeyError as e:
                msg = (
                    f"{e.args[0]} Make sure you have validation data if you use "
                    "validation scores for checkpointing.")
                raise SkorchException(msg)

        if self.event_name is not None:
            net.history.record(self.event_name, bool(do_checkpoint))

        if do_checkpoint:
            self.save_model(net)
            self._sink("A checkpoint was triggered in epoch {}.".format(
                len(net.history) + 1
            ), net.verbose)
        self.save_history(net)

    def _f_kwargs(self):
        return {key: getattr(self, key) for key in dir(self)
                if key.startswith('f_') and (key != 'f_history_')}

    def save_history(self, net):
        kwargs_module, kwargs_other = _check_f_arguments(
            self.__class__.__name__, **self._f_kwargs())
        f_history = kwargs_other.get('f_history')
        if f_history is not None:
            f = self.f_history_
            self._save_params(f, net, "f_history", "history")

    def save_model(self, net):
        """Save the model.

        This function saves some or all of the following:

          - model parameters;
          - optimizer state;
          - criterion state;
          - training history;
          - custom modules;
          - entire model object.

        """
        kwargs_module, kwargs_other = _check_f_arguments(
            self.__class__.__name__, **self._f_kwargs())

        for key, val in kwargs_module.items():
            if val is None:
                continue

            f = self._format_target(net, val, -1)
            key = key[:-1]  # remove trailing '_'
            self._save_params(f, net, 'f_' + key, key + " state")

        f_pickle = kwargs_other.get('f_pickle')
        if f_pickle:
            f_pickle = self._format_target(net, f_pickle, -1)
            with open_file_like(f_pickle, 'wb') as f:
                pickle.dump(net, f)

    @property
    def f_history_(self):
        # This is a property and not in initialize to allow ``NeuralNet``
        # to call ``load_params`` without needing the checkpoint to
        # by initialized.
        if self.f_history is None:
            return None
        return os.path.join(
            self.dirname, self.fn_prefix + self.f_history)

    def get_formatted_files(self, net):
        """Returns a dictionary of formatted filenames"""
        idx = -1
        if (
                self.event_name is not None and
                net.history
        ):
            for i, v in enumerate(net.history[:, self.event_name]):
                if v:
                    idx = i

        return {key: self._format_target(net, val, idx) for key, val
                in self._f_kwargs().items()}

    def _save_params(self, f, net, f_name, log_name):
        try:
            net.save_params(
                **{f_name: f, 'use_safetensors': self.use_safetensors})
        except Exception as e:  # pylint: disable=broad-except
            self._sink(
                "Unable to save {} to {}, {}: {}".format(
                    log_name, f, type(e).__name__, e), net.verbose)

    def _format_target(self, net, f, idx):
        """Apply formatting to the target filename template."""
        if f is None:
            return None
        if isinstance(f, str):
            f = self.fn_prefix + f.format(
                net=net,
                last_epoch=net.history[idx],
                last_batch=net.history[idx, 'batches', -1],
            )
            return os.path.join(self.dirname, f)
        return f

    def _validate_filenames(self):
        """Checks if passed filenames are valid.

        Specifically, f_* parameter should not be passed in
        conjunction with dirname.

        """
        _check_f_arguments(self.__class__.__name__, **self._f_kwargs())

        if not self.dirname:
            return

        def _is_truthy_and_not_str(f):
            return f and not isinstance(f, str)

        if any(_is_truthy_and_not_str(val) for val in self._f_kwargs().values()):
            raise SkorchException(
                'dirname can only be used when f_* are strings')

    def _sink(self, text, verbose):
        #  We do not want to be affected by verbosity if sink is not print
        if (self.sink is not print) or verbose:
            self.sink(text)


def net_def(
    net_class: nn.Module, net_name,
    classes, classifier_kwargs: dict,
):
    net_name = net_name + '_'

    classifier_kwargs.setdefault('lr', 0.001)
    classifier_kwargs.setdefault('criterion', nn.CrossEntropyLoss)
    classifier_kwargs.setdefault('batch_size', 64)
    classifier_kwargs.setdefault('optimizer', optim.SGD)
    # classifier_kwargs.setdefault('optimizer', optim.Adam)
    classifier_kwargs.setdefault('optimizer__momentum', 0.92)
    # classifier_kwargs.setdefault(
    classifier_kwargs.setdefault('iterator_train__shuffle', True)
    classifier_kwargs.setdefault('iterator_train__num_workers', 6)
    classifier_kwargs.setdefault('iterator_valid__num_workers', 6)
    classifier_kwargs.setdefault('device', get_device())
    default_callbacks = [
        EpochScoring(scoring='f1_macro', name='valid_f1',
                     lower_is_better=False),
        # LRScheduler(policy='StepLR', step_size=7, gamma=0.1),
        MyCheckpoint(monitor='valid_f1_best', load_best=True,
                     fn_prefix=os.path.join('weights', net_name))
    ]
    if classifier_kwargs.get('callbacks') is not None:
        default_callbacks.extend(classifier_kwargs['callbacks'])
    classifier_kwargs['callbacks'] = default_callbacks

    if classifier_kwargs.get('train_split') is not None:
        classifier_kwargs['train_split'] = predefined_split(
            classifier_kwargs.get('train_split'))

    net = NeuralNetClassifier(
        net_class,
        classes=classes,
        warm_start=True,              # continue the last fit
        **classifier_kwargs
    )

    setattr(net, 'net_name', net_name)
    return net


def net_fit(net: skorch.NeuralNet, x, y, epochs):
    if os.path.exists(os.path.join('weights', f"{net.net_name}params.pt")):
        with open(os.path.join('weights', f"{net.net_name}history.json"), 'rb') as f:
            h = json.load(f)
        if net.history != h:
            print(f"Load saved params: {net.net_name}params.pt")
            net.initialize()
            pms = {}
            pms['f_params'] = os.path.join(
                'weights', f"{net.net_name}params.pt")
            pms['f_history'] = os.path.join(
                'weights', f"{net.net_name}history.json")
            if os.path.exists(os.path.join('weights', f"{net.net_name}optimizer.pt")):
                pms['f_optimizer'] = os.path.join(
                    'weights', f"{net.net_name}optimizer.pt")
            if os.path.exists(os.path.join('weights', f"{net.net_name}optimizer.pt")):
                pms['f_criterion'] = os.path.join(
                    'weights', f"{net.net_name}criterion.pt")
            net.load_params(**pms)
        else:
            print(f"Current params are the latest, continue the training.")

        print('Histories:')
        p = PrintLog()
        p.initialize()
        data = net.history[-1]
        verbose = net.verbose
        tabulated = p.table(data)

        header, lines = tabulated.split('\n', 2)[:2]
        p._sink(header, verbose)
        p._sink(lines, verbose)

        for h in net.history:
            tabulated = p.table(h)
            header, lines = tabulated.split('\n', 2)[:2]
            p._sink(tabulated.split('\n')[2], verbose)
            if p.sink is print:
                sys.stdout.flush()

        print('\n\n', 'New training:')

    net.fit(X=x, y=y, epochs=epochs)
    # print(net.history)
