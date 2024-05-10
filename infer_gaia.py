
import os
import pickle
import sys
import shutil
import h5py

import numpy as np
import scipy.special as special
import seaborn as sns
import pytorch_lightning.loggers as pl_loggers
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from memory_profiler import profile

import datasets
from models import models, classifier, utils, infer_utils


def read_dataset(data_fn, features):
    """ Read in the dataset from the hdf5 file. """
    x = []
    with h5py.File(data_fn, 'r') as f:
        # if radial velocity is in the features, get only the stars with
        # availabel radial velocity
        if 'radial_velocity' in features:
            mask = ~np.isnan(f['radial_velocity'][:])
        else:
            mask = np.ones(len(f['source_id']), dtype=bool)
        for feature in features:
            x.append(f[feature][mask])
        source_id = f['source_id'][mask]

    x = np.stack(x, axis=-1) if len(x) > 0 else None
    y = np.zeros(len(x), dtype=bool)
    index = np.arange(len(mask))[mask]

    return x, y, source_id, index

def infer():

    # read in the model
    logdir = '/ocean/projects/phy210068p/tvnguyen/accreted_catalog/logging'
    name = 'small-spine-6'
    checkpoint = 'epoch=24-step=1194775.ckpt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # read in the dataset and prepare the data loader for training
    data_root = '/ocean/projects/phy210068p/tvnguyen/accreted_catalog/datasets'
    data_name = 'GaiaDR3_transfer'
    data_features = ['ra', 'dec', 'pmra', 'pmdec', 'parallax', 'radial_velocity']
    data_dir = os.path.join(data_root, data_name)

    # catalog name
    catalog_root = '/ocean/projects/phy210068p/shared/gaia_catalogs'
    catalog_name = 'GaiaDR3_FeH_reduced_v4'

    checkpoint_path = os.path.join(logdir, name, 'lightning_logs/checkpoints', checkpoint)
    model = classifier.MLPClassifier.load_from_checkpoint(
        checkpoint_path, map_location=device)


    for i in range(10):

        data_fn = os.path.join(data_dir, f'data.{i}.hdf5')
        print(f'Reading data from {data_fn}')

        features, labels, source_id, index = read_dataset(data_fn, data_features)
        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        data_loader = DataLoader(
            TensorDataset(features, labels), batch_size=8196, shuffle=False)

        # inference
        y_pred, y_true = infer_utils.infer(
            model, data_loader, softmax=False, to_numpy=True)
        y_pred_score = special.softmax(y_pred, axis=1)
        y_pred_score = y_pred_score[..., 1]

        # save the catalog
        catalog_path = os.path.join(catalog_root, catalog_name, f'labels.{i}.hdf5')
        os.makedirs(os.path.dirname(catalog_path), exist_ok=True)
        with h5py.File(catalog_path, 'w') as f:
            f.create_dataset('score', data=y_pred_score)
            f.create_dataset('source_id', data=source_id)
            f.create_dataset('index', data=index)

    return

if __name__ == '__main__':
    infer()