
import os
import pickle
import sys
import shutil

import ml_collections
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
from torch.utils.data import DataLoader, TensorDataset
from absl import flags, logging
from ml_collections import config_flags
from memory_profiler import profile

import datasets
from models import models, classifier, utils

logging.set_verbosity(logging.INFO)

@profile
def train(
    config: ml_collections.ConfigDict, workdir: str = "./logging/"
):
    # set up work directory
    if not hasattr(config, "name"):
        name = utils.get_random_name()
    else:
        name = config["name"]
    logging.info("Starting training run {} at {}".format(name, workdir))

    # set up random seed
    pl.seed_everything(config.seed)

    workdir = os.path.join(workdir, name)
    checkpoint_path = None
    if os.path.exists(workdir):
        if config.overwrite:
            shutil.rmtree(workdir)
        elif config.get('checkpoint', None) is not None:
            checkpoint_path = os.path.join(
                workdir, 'lightning_logs/checkpoints', config.checkpoint)
        else:
            raise ValueError(
                f"Workdir {workdir} already exists. Please set overwrite=True "
                "to overwrite the existing directory.")
    else:
        os.makedirs(workdir)

    # copy the config to the workdir
    with open(os.path.join(workdir, "config.yaml"), "w") as f:
        f.write(config.to_yaml())

    # read in the dataset and prepare the data loader for training
    data_dir = os.path.join(config.data.root, config.data.name)

    train_loader, val_loader, norm_dict, class_weights = datasets.prepare_dataloader(
        data_dir, config.data.features,
        num_datasets=config.data.get("num_datasets", 1),
        subsample_factor=config.data.get("subsample_factor", 1),
        class_weight_scales=config.class_weight_scales,
        train_frac=config.train_frac,
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        seed=config.seed,  # reset seed for splitting train/val
    )

    # create model
    class_weights = config.get('class_weights') or class_weights
    logging.info("Using class weights: {}".format(class_weights))

    model = classifier.MLPClassifier(
        config.model.input_dim, config.model.output_dim,
        hidden_sizes=config.model.hidden_sizes,
        activation_args=config.model.activation,
        optimizer_args=config.optimizer,
        scheduler_args=config.scheduler,
        class_weights=class_weights,
        transfer_layers=config.model.get("transfer_layers", None),
        norm_dict=norm_dict,
    )

    # create the trainer object
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor=config.monitor, patience=config.patience, mode=config.mode,
            verbose=True),
        pl.callbacks.ModelCheckpoint(
            monitor=config.monitor, save_top_k=config.save_top_k,
            mode=config.mode, save_weights_only=False),
        pl.callbacks.LearningRateMonitor("epoch"),
    ]

    train_logger = pl_loggers.TensorBoardLogger(workdir, version='')
    trainer = pl.Trainer(
        default_root_dir=workdir,
        max_epochs=config.num_epochs,
        accelerator=config.accelerator,
        callbacks=callbacks,
        logger=train_logger,
        enable_progress_bar=config.get("enable_progress_bar", True),
    )

    # train the model
    logging.info("Training model...")
    trainer.fit(
        model, train_loader, val_loader,
        ckpt_path=checkpoint_path)

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        "config",
        None,
        "File path to the training or sampling hyperparameter configuration.",
        lock_config=True,
    )
    # Parse flags
    FLAGS(sys.argv)

    # Start training run
    train(config=FLAGS.config, workdir=FLAGS.config.workdir)
