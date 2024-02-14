
import os
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from . import io_utils, preprocess_utils


def read_process_dataset(
    data_dir: Union[str, Path], features: List[str],
    num_datasets: int = 1, subsample_factor: int = 1
):
    """ Read the dataset and preprocess

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the stream data.
    features : list of str
        List of features to use for the regression.
    num_datasets : int, optional
        Number of datasets to read in. Default is 1.
    subsample_factor : int, optional
        Factor to subsample the dataset. Default is 1.
    """
    x, y  = [], []

    for i in range(num_datasets):
        data_fn = os.path.join(data_dir, f'data.{i}.hdf5')

        if os.path.exists(data_fn):
            print('Reading in data from {}'.format(data_fn))
        else:
            print('Dataset {} not found. Skipping...'.format(i))
            continue

        # read in the data and label
        data = io_utils.read_dataset(data_fn, unpack=True)

        # TODO: add the rest of the code

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)

    return x, y


def prepare_dataloader(
    data: Tuple,
    norm_dict: dict = None,
    train_frac: float = 0.8,
    train_batch_size: int = 128,
    eval_batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 42
):
    """ Create dataloaders for training and evaluation. """
    pl.seed_everything(seed)

    # unpack the data and convert to tensor
    x, y = data
    num_train = int(train_frac * len(x))
    shuffle = np.random.permutation(len(x))
    x = torch.tensor(x[shuffle], dtype=torch.float32)
    y = torch.tensor(y[shuffle], dtype=torch.float32)  # TODO: check if this is the correct dtype

    # normalize the data
    if norm_dict is None:
        x_loc = x[:num_train].mean(dim=0)
        x_scale = x[:num_train].std(dim=0)
        norm_dict = {"x_loc": x_loc, "x_scale": x_scale,}
    else:
        x_loc = norm_dict["x_loc"]
        x_scale = norm_dict["x_scale"]
    x = (x - x_loc) / x_scale

    # create data loader
    train_dset = TensorDataset(x[:num_train], y[:num_train])
    val_dset = TensorDataset(x[num_train:], y[num_train:])
    train_loader = DataLoader(
        train_dset, batch_size=train_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(
        val_dset, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, norm_dict
