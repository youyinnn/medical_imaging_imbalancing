
from lightning.pytorch.tuner import Tuner
from typing import Dict, Type
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.callbacks import Callback
import lightning as L
from torchvision import models
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, top_k_accuracy_score
import timm


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

    def before_fit(self):
        if self.model.lr < 0:
            print('Learning rate is not set, use tuner to find it.')
            trainer = L.Trainer(
                # disable logger of the tuner trainer
                max_epochs=50, enable_checkpointing=False, logger=False)
            tuner = Tuner(trainer)
            # start with a valid lr
            dm = self.datamodule
            dm.setup('fit')
            iterations = len(dm.train_dataloader())
            self.model.lr = 1e-3
            lr_finder = tuner.lr_find(
                self.model, attr_name="lr", datamodule=self.datamodule,
                num_training=int(
                    iterations * self.config.fit.lr_tuner.num_training_multiple),
                # this param has bug, set to None
                early_stop_threshold=None)
            print('Tuned learning rate:', self.model.lr)
            self.config.fit.model.lr = self.model.lr


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
        self.prog_bar = prog_bar
        self.output_features = output_features

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
        }, prog_bar=self.prog_bar)

    def training_step(self, batch, batch_idx):
        x, y = batch
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
        optimizer = self.optimizer_class(
            self.model.parameters(), lr=self.lr, **self.optimizer_kwargs)
        return optimizer
