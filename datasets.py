
import os
import h5py
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def read_dataset(data_fn: str, features: List[str] = None):
    """ Read in the dataset from the hdf5 file. """
    x = []
    with h5py.File(data_fn, 'r') as f:
        y = f['is_accreted'][:]
        if features is not None:
            for feature in features:
                x.append(f[feature][:])
    x = np.stack(x, axis=-1) if len(x) > 0 else None
    return x, y


def read_process_datasets(
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
        x_i, y_i = read_dataset(data_fn, features)
        x.append(x_i)
        y.append(y_i)

    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)

    # subsample the data
    if subsample_factor > 1:
        x = x[::subsample_factor]
        y = y[::subsample_factor]

    return x, y


def prepare_dataloader(
    data_dir: Union[str, Path],
    features: List[str],
    num_datasets: int = 1,
    subsample_factor: int = 1,
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
    x, y = read_process_datasets(
        data_dir, features, num_datasets, subsample_factor)
    shuffle = np.random.permutation(len(x))
    x = x[shuffle]
    y = y[shuffle]
    num_train = int(train_frac * len(x))

    # compute class weights
    n0 = numpy.count_nonzero(y==0)
    n1 = numpy.count_nonzero(y==1)
    w0 = (n1 + n0) / n0
    w1 = (n1 + n0) / n1
    class_weights = [w0, w1]

    # normalize the data
    if norm_dict is None:
        x_loc = x[:num_train].mean(axis=0)
        x_scale = x[:num_train].std(axis=0)
        norm_dict = {"x_loc": x_loc, "x_scale": x_scale,}
    else:
        x_loc = norm_dict["x_loc"]
        x_scale = norm_dict["x_scale"]
    # NOTE: Do not change this to x = (x-x_loc) / x_scale because it's not in-place
    # and cost more memory
    x -= x_loc
    x /= x_scale

    # convert to tensor
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # create data loader
    train_dset = TensorDataset(x[:num_train], y[:num_train])
    val_dset = TensorDataset(x[num_train:], y[num_train:])
    train_loader = DataLoader(
        train_dset, batch_size=train_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(
        val_dset, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, norm_dict, class_weights

