
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl

from . import models, models_utils


class MLPClassifier(pl.LightningModule):
    def __init__(
        self, input_dim: int, output_dim: int,
        hidden_sizes: List[int] = [512],
        activation_args: Optional[Dict[str, Any]] = None,
        optimizer_args: Optional[Dict[str, Any]] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        class_weights: Optional[List[float]] = None,
        norm_dict: Optional[Dict[str, Any]] = None,
        transfer_layers: Optional[List[int]] = None
    ):
        """
        Parameters
        ----------
        input_dim : int
            The size of the input
        output_dim: int
            The size of the output
        hidden_sizes : list of int, optional
            The sizes of the hidden layers. Default: [512]
        activation_args: dict, optional
            Arguments for the activation function. Default: None
        optimizer_args : dict, optional
            Arguments for the optimizer. Default: None
        scheduler_args : dict, optional
            Arguments for the scheduler. Default: None
        class_weights: list, optional
            The class weights. Default: None
        norm_dict : dict, optional
            The normalization dictionary. For bookkeeping purposes only.
            Default: None
        transfer_layers: list of int, optional
            The layers to freeze. Default: None
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes
        self.activation_args = activation_args or {}
        self.optimizer_args = optimizer_args or {}
        self.scheduler_args = scheduler_args or {}
        self.class_weights = class_weights
        self.norm_dict = norm_dict
        self.transfer_layers = transfer_layers
        self.save_hyperparameters()

        self._setup_model()

    def _setup_model(self):
        # create the featurizer
        activation_fn = models_utils.get_activation(self.activation_args)
        self.featurizer = models.MLP(
            self.input_dim, self.output_dim, hidden_sizes=self.hidden_sizes, activation_fn=activation_fn,
            transfer_layers=self.transfer_layers
        )

        if self.class_weights is not None:
            weight = torch.tensor(self.class_weights, dtype=torch.float32)
        else:
            weight = None
        self.loss_fn = nn.CrossEntropyLoss(weight=weight)

    def _prepare_training_batch(self, batch):
        """ Prepare the batch for training. """
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        # normalize the inputs
        x_loc = torch.tensor(
            self.norm_dict['x_loc'], dtype=torch.float32, device=self.device)
        x_scale = torch.tensor(
            self.norm_dict['x_scale'], dtype=torch.float32, device=self.device)
        x = (x - x_loc) / x_scale

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
        batch_dict = self._prepare_training_batch(batch)
        x = batch_dict['x']
        y = batch_dict['y']
        batch_size = len(x)

        # forward pass
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)

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
        batch_dict = self._prepare_training_batch(batch)
        x = batch_dict['x']
        y = batch_dict['y']
        batch_size = len(x)

        # forward pass
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)

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
