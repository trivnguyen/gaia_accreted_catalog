
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl

from . import models, models_utils, flows_utils


class MLPClassifier(pl.LightningModule):
    def __init__(
        self, input_size: int, output_size: int,
        hidden_sizes: List[int] = [512], activation: str = 'relu',
        optimizer_args: Optional[Dict[str, Any]] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        norm_dict: Optional[Dict[str, Any]] = None
    ):
        """
        Parameters
        ----------
        input_size : int
            The size of the input
        output_size: int
            The size of the output
        hidden_sizes : list of int, optional
            The sizes of the hidden layers. Default: [512]
        activation : str, optional
            The activation function to use. Default: 'relu'
        optimizer_args : dict, optional
            Arguments for the optimizer. Default: None
        scheduler_args : dict, optional
            Arguments for the scheduler. Default: None
        norm_dict : dict, optional
            The normalization dictionary. For bookkeeping purposes only.
            Default: None
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.optimizer_args = optimizer_args or {}
        self.scheduler_args = scheduler_args or {}
        self.norm_dict = norm_dict
        self.save_hyperparameters()

        self._setup_model()

    def _setup_model(self):
        # create the featurizer
        activation_fn = models_utils.get_actizvation(
            self.activation_fn)
        self.featurizer = models.MLP(
            self.input_size, self.output_size, hidden_sizes=self.hidden_sizes,
            activation_fn=activation_fn
        )

    def _prepare_training_batch(self, batch):
        """ Prepare the batch for training. """
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        # return a dictionary of the inputs
        return_dict = {
            'x': x,
            'y': y,
        }
        return return_dict

    def forward(self, x):
        x = self.featurizer(x)
        return x

    def training_step(self, batch, batch_idx):
        # prepare the batch
        batch_dict = self._prepare_batch(batch)
        x = batch_dict['x']
        y = batch_dict['y']
        batch_size = len(x)

        # forward pass
        y_hat = self.forward(x)
        loss = torch.nn.CrossEntropyLoss()(y_hat, y)

        # calculate the accuracy
        acc = y.eq(y_hat.argmax(dim=1)).float().mean()

        # log the loss
        self.log(
            'train_loss', loss, on_step=True, on_epoch=True, logger=True,
            prog_bar=True, batch_size=batch_size)
        self.log(
            'train_acc', acc, on_step=True, on_epoch=True, logger=True,
            prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        # prepare the batch
        batch_dict = self._prepare_batch(batch)
        x = batch_dict['x']
        y = batch_dict['y']
        batch_size = len(x)

        # forward pass
        y_hat = self.forward(x)
        loss = torch.nn.CrossEntropyLoss()(y_hat, y)

        # calculate the accuracy
        acc = y.eq(y_hat.argmax(dim=1)).float().mean()

        # log the loss
        self.log(
            'val_loss', loss, on_step=True, on_epoch=True, logger=True,
            prog_bar=True, batch_size=batch_size)
        self.log(
            'val_acc', acc, on_step=True, on_epoch=True, logger=True,
            prog_bar=True, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        """ Initialize optimizer and LR scheduler """
        return models_utils.configure_optimizers(
            self.parameters(), self.optimizer_args, self.scheduler_args)
