import json
import os
import sys
import skorch
import time
import torch
import torch.nn as nn
import numpy as np
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
from skorch.dataset import unpack_data
from skorch.utils import to_tensor


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


class ResetedSkorchLRScheduler(LRScheduler):

    def __init__(self, checkpoint_monitor, lr_reset_monitor_list,
                 max_iter, monitors_indicators=None,
                 policy='WarmRestartLR', monitor='train_loss',
                 event_name="event_lr", step_every='epoch', **kwargs):

        self.checkpoint_monitor = checkpoint_monitor
        self.lr_reset_monitor_list = lr_reset_monitor_list
        # true if the higher the better
        self.monitors_indicators = monitors_indicators
        self.max_iter = max_iter
        self.current_count = 0
        self.resetor_name = str(time.time())
        vars(self).update(kwargs)
        super().__init__(policy, monitor, event_name, step_every, **kwargs)

        higher_better = ['acc', 'f1', 'precision', 'recall', 'auc']
        lower_better = ['loss']

        if self.monitors_indicators is None:
            self.monitors_indicators = []
            for reset_monitor in lr_reset_monitor_list:
                lower_ = any(lbm in reset_monitor for lbm in lower_better)
                higher_ = any(hbm in reset_monitor for hbm in higher_better)
                indicator = lower_ ^ higher_
                # if not show in both, then the higher the better as default
                self.monitors_indicators.append(
                    True if not indicator else higher_)

    @property
    def kwargs(self):
        # These are the parameters that are passed to the
        # scheduler. Parameters that don't belong there must be
        # excluded.
        excluded = (
            'checkpoint_monitor', 'lr_reset_monitor_list',
            'max_iter', 'monitors_indicators', 'current_count',
            'resetor_name',
            'policy', 'monitor', 'event_name', 'step_every'
        )
        kwargs = {key: val for key, val in vars(self).items()
                  if not (key in excluded or key.endswith('_'))}
        return kwargs

    def not_improved(self, net: skorch.NeuralNet):
        checkpoint_record_idx = int(np.argmax(
            net.history[:, self.checkpoint_monitor]))
        if len(net.history) > self.max_iter and len(net.history) > (checkpoint_record_idx + 1):
            ckp_monitor_over_reset = net.history[checkpoint_record_idx,
                                                 self.lr_reset_monitor_list]
            last_monitor = net.history[-1, self.lr_reset_monitor_list]

            decision = []
            for i, monitors_indicator in enumerate(self.monitors_indicators):
                rs = ckp_monitor_over_reset[i] >= last_monitor[i]
                decision.append(rs if monitors_indicator else not rs)

            return np.all(decision)
        return False

    def on_epoch_end(self, net: skorch.NeuralNet, **kwargs):
        if self.not_improved(net):
            self.current_count += 1
        else:
            self.current_count = 0

        if len(net.history) > self.max_iter and self.current_count > 0:
            net.history.record(f"event_not_imp_count", self.current_count)
        else:
            net.history.record(f"event_not_imp_count", '')

        if self.current_count == self.max_iter:
            old_step_count = self.lr_scheduler_._step_count
            net.initialize_optimizer()
            self.lr_scheduler_ = self._get_scheduler(
                net, self.policy_, **self.kwargs
            )
            self.lr_scheduler_._step_count = old_step_count
            net.history.record(f"{self.event_name}_reset", 'x')
            self.current_count = 0
        else:
            net.history.record(f"{self.event_name}_reset", ' ')
        super().on_epoch_end(net, **kwargs)


class CutMixedNeuralNetClassifier(NeuralNetClassifier):

    def get_loss(self, y_pred, y_true, X=None, training=False):
        y_true = to_tensor(y_true, device=self.device)
        if training:
            mixed_img, picked_label, lam = X
            picked_label = to_tensor(picked_label, device=self.device)
            lam = lam.to(self.device)

            # loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
            al = self.criterion_(y_pred, y_true)
            bl = self.criterion_(y_pred, picked_label)
            return ((al * lam) + (bl * (1 - lam))).mean()
        else:
            return self.criterion_(y_pred, y_true).mean()

    def train_step_single(self, batch, **fit_params):
        self._set_training(True)
        Xi, yi = unpack_data(batch)
        y_pred = self.infer(Xi[0], **fit_params)
        loss = self.get_loss(y_pred, yi, X=Xi, training=True)
        loss.backward()
        return {
            'loss': loss,
            'y_pred': y_pred,
        }


def net_def(
    net_class: nn.Module, net_name,
    classes, classifier_kwargs: dict,
    cut_mixed=False
):
    net_name = net_name + '_'

    classifier_kwargs.setdefault('lr', 0.001)
    classifier_kwargs.setdefault('criterion', nn.CrossEntropyLoss)
    classifier_kwargs.setdefault('batch_size', 64)
    classifier_kwargs.setdefault('optimizer', optim.SGD)
    # classifier_kwargs.setdefault('optimizer', optim.Adam)
    classifier_kwargs.setdefault('optimizer__momentum', 0.9)
    classifier_kwargs.setdefault('iterator_train__shuffle', True)
    classifier_kwargs.setdefault('iterator_train__num_workers', 6)
    classifier_kwargs.setdefault('iterator_valid__num_workers', 6)
    classifier_kwargs.setdefault('device', get_device())

    cp_fn_prefix = os.path.join('weights', net_name, net_name)
    default_callbacks = [
        EpochScoring(scoring='f1_macro', name='valid_f1',
                     lower_is_better=False),
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


def net_fit(net: skorch.NeuralNet, x, y, max_epochs):
    weight_path_pre = os.path.join('weights', net.net_name)
    if not os.path.exists(weight_path_pre):
        os.makedirs(weight_path_pre, exist_ok=True)

    params_pt_path = os.path.join(
        weight_path_pre, f"{net.net_name}params.pt")
    history_json_path = os.path.join(
        weight_path_pre, f"{net.net_name}history.json")
    optimizer_pt_path = os.path.join(
        weight_path_pre, f"{net.net_name}optimizer.pt")
    criterion_pt_path = os.path.join(
        weight_path_pre, f"{net.net_name}criterion.pt")
    if os.path.exists(params_pt_path):
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
        if epochs > 0:
            print('\n\n', 'Continue training:')
            net.partial_fit(X=x, y=y, epochs=epochs)
    else:
        epochs = max_epochs
        if epochs > 0:
            print('\n\n', 'New training:')
            net.fit(X=x, y=y, epochs=epochs)

    # print(net.history)
    return net
