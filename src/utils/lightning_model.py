
import logging
import os
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.cli import LightningCLI
import lightning as L
from torchvision import models
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, top_k_accuracy_score
import timm
from timm.scheduler import tanh_lr, cosine_lr, plateau_lr, step_lr
from torchvision.transforms import v2
from utils.pytorch_helper import X_Aug


def pytorch_resnet_fine_tuning(model, output_features):
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, output_features)
    return model


model_map = dict(
    pytorch_resnet50=dict(
        model_class=models.resnet50,
        weights=models.ResNet50_Weights.IMAGENET1K_V1,
        fine_tune=pytorch_resnet_fine_tuning
    ),
    timm_mixer_b16_224_in21k=dict(
        model_class=timm.create_model,
        weights='mixer_b16_224_in21k',
    )
)


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--lr_tuner.num_training_multiple", default=1)

    def tuner_(self, subcommand: str):
        if self.model.lr < 0:
            print('Learning rate is not set, use tuner to find it.')

            # disable warning: https://github.com/Lightning-AI/pytorch-lightning/issues/3431
            log_level_before = logging.getLogger(
                "lightning.pytorch.utilities.rank_zero").getEffectiveLevel()
            logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(
                logging.WARNING)
            trainer = L.Trainer(
                # disable logger of the tuner trainer
                max_epochs=50, enable_checkpointing=False, logger=False, enable_model_summary=False)
            tuner = Tuner(trainer)
            # start with a valid lr
            dm = self.datamodule
            dm.setup(subcommand)
            iterations = len(dm.train_dataloader())
            self.model.lr = 1e-3
            ns = getattr(self.config, subcommand)
            lr_finder = tuner.lr_find(
                self.model, attr_name="lr", datamodule=self.datamodule,
                num_training=int(
                    iterations * ns.lr_tuner.num_training_multiple),
                # this param has bug, set to None
                early_stop_threshold=None)
            ns.model.lr = self.model.lr
            logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(
                log_level_before)

    def before_fit(self):
        self.get_autodl_tk()
        self.tuner_(subcommand='fit')

    def before_validate(self):
        self.tuner_(subcommand='validate')

    def get_autodl_tk(self):
        from pathlib import Path
        home = str(Path.home())
        tk_path = os.path.join(home, 'autodl_tk')
        if os.path.exists(tk_path) and not self.config.fit.trainer.fast_dev_run:
            print("Token exist")
            return tk_path

    def after_fit(self):
        tk_path = self.get_autodl_tk()
        if tk_path is not None and not self.config.fit.trainer.fast_dev_run:
            with open(tk_path, 'r') as f:
                import requests
                headers = {"Authorization": f.read()}
                resp = requests.post("https://www.autodl.com/api/v1/wechat/message/send",
                                     json={
                                         "title": "MII",
                                         "name": self.config.fit.trainer.default_root_dir,
                                         "content": "Training finshed."
                                     }, headers=headers)


class LightningClassifier(L.LightningModule):

    def __init__(self,
                 model_key: str,
                 output_features: int,
                 lr: float,
                 prog_bar: bool = False,
                 loss_fn_key='CrossEntropyLoss',
                 loss_fn_kwargs: dict = {},
                 optimizer_key='Adam',
                 optimizer_kwargs: dict = {},
                 lr_scheduler_key=None,
                 lr_scheduler_base='torch',
                 lr_scheduler_kwargs: dict = None,
                 lr_scheduler_config: dict = None,
                 cutmix_or_mixup_alpha: float = 1,
                 cutmix_or_mixup_prob: float = 0.0,
                 cutmix_only: bool = False,
                 x_aug_prob: float = 0.0,
                 x_aug_kwargs: dict = {},
                 pretrained: bool = True):
        super().__init__()

        model_info = model_map[model_key]
        model_class = model_info['model_class']
        model_weights = model_info['weights']

        if model_key.startswith('pytorch'):
            model = model_class(weights=model_weights if pretrained else None)
            model = model_info['fine_tune'](model, output_features)
        if model_key.startswith('timm_'):
            model = model_class(model_weights, pretrained=pretrained,
                                num_classes=output_features)

        self.model = model
        self.lr = lr
        self.loss_fn_key = loss_fn_key
        self.loss_fn_kwargs = loss_fn_kwargs
        self.loss_fn = getattr(torch.nn, loss_fn_key)(**loss_fn_kwargs)

        self.optimizer_key = optimizer_key
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_class = getattr(torch.optim, optimizer_key)

        self.lr_scheduler_key = lr_scheduler_key
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.lr_scheduler_config = lr_scheduler_config
        self.lr_scheduler_base = lr_scheduler_base
        if lr_scheduler_base == 'torch':
            lrs_base = torch.optim.lr_scheduler
            self.lr_scheduler_class = getattr(
                lrs_base, lr_scheduler_key) if lr_scheduler_key is not None else None
        elif lr_scheduler_base.startswith('timm'):
            if lr_scheduler_key == 'CosineLRScheduler':
                self.lr_scheduler_class = cosine_lr.CosineLRScheduler
            if lr_scheduler_key == 'TanhLRScheduler':
                self.lr_scheduler_class = tanh_lr.TanhLRScheduler
            if lr_scheduler_key == 'StepLRScheduler':
                self.lr_scheduler_class = step_lr.StepLRScheduler
            if lr_scheduler_key == 'PlateauLRScheduler':
                self.lr_scheduler_class = plateau_lr.PlateauLRScheduler

            if self.lr_scheduler_config is None:
                self.lr_scheduler_config = {
                    # "interval": "epoch",
                    # How many epochs/steps should pass between calls to
                    # `scheduler.step()`. 1 corresponds to updating the learning
                    # rate after every epoch/step.
                    # "frequency": 1,
                    # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                    # "monitor": "val_loss",
                    # If set to `True`, will enforce that the value specified 'monitor'
                    # is available when the scheduler is updated, thus stopping
                    # training if not found. If set to `False`, it will only produce a warning
                    # "strict": True,
                    # If using the `LearningRateMonitor` callback to monitor the
                    # learning rate progress, this keyword can be used to specify
                    # a custom logged name
                    # "name": None,
                }

        self.prog_bar = prog_bar
        self.output_features = output_features

        self.x_aug_prob = x_aug_prob
        self.x_aug_kwargs = x_aug_kwargs
        if self.x_aug_prob > 0:
            self.x_aug = X_Aug(self.model, **x_aug_kwargs)

        self.cutmix_or_mixup_prob = cutmix_or_mixup_prob
        self.cutmix_only = cutmix_only
        if self.cutmix_or_mixup_prob > 0:
            cutmix = v2.CutMix(num_classes=output_features,
                               alpha=cutmix_or_mixup_alpha)
            if not self.cutmix_only:
                mixup = v2.MixUp(num_classes=output_features,
                                 alpha=cutmix_or_mixup_alpha)
                self.cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
            else:
                self.cutmix_or_mixup = cutmix

    def forward(self, x):
        return self.model(x)

    def eval_metrics(self, event_key, batch, batch_idx, output, loss):
        x, y = batch
        yy = y.cpu().numpy()
        output = output.detach()
        oo = torch.argmax(output, dim=1).cpu().numpy()
        f1 = f1_score(yy, oo, average='macro')
        acc = accuracy_score(yy, oo)
        top_5 = top_k_accuracy_score(yy, output.cpu(), k=5,
                                     labels=[i for i in range(self.output_features)])
        self.log_dict({
            f"{event_key}_loss": loss,
            f"{event_key}_f1": f1,
            f"{event_key}_acc": acc,
            f"{event_key}_top5": top_5,
            f"{event_key}_lr": self.optimizer.param_groups[0]['lr'],
        }, prog_bar=self.prog_bar)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.cutmix_or_mixup_prob > 0.0 and \
                torch.rand(1).item() >= self.cutmix_or_mixup_prob:
            x, y = self.cutmix_or_mixup(x, y)
        if self.x_aug_prob > 0.0 and \
                torch.rand(1).item() >= self.x_aug_prob:
            x, y = self.x_aug(batch)
        output = self.model(x)
        loss = self.loss_fn.to(x.device)(output, y)
        self.eval_metrics('train', batch, batch_idx, output, loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        loss = self.loss_fn.to(x.device)(output, y)
        self.eval_metrics('test', batch, batch_idx, output, loss)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        loss = self.loss_fn.to(x.device)(output, y)
        self.eval_metrics('val', batch, batch_idx, output, loss)

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(
        #     self.parameters(), lr=self.lr, momentum=0.9)
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.optimizer = self.optimizer_class(
            self.model.parameters(), lr=self.lr, **self.optimizer_kwargs)
        if self.lr_scheduler_class is not None:
            # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers
            self.lr_scheduler = self.lr_scheduler_class(
                self.optimizer, **self.lr_scheduler_kwargs)
            return dict(
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                ** self.lr_scheduler_config
            )
        else:
            return self.optimizer

    def lr_scheduler_step(self, scheduler, metric):
        if self.lr_scheduler_base == 'timm':
            # timm's scheduler need the epoch value
            scheduler.step(epoch=self.current_epoch)
        else:
            scheduler.step()
