
from ml_collections import config_dict

def get_config():
    cfg = config_dict.ConfigDict()

    cfg.seed = 42

    # data configuration
    cfg.data = config_dict.ConfigDict()
    cfg.data.root = '/ocean/projects/phy210068p/tvnguyen/accreted_catalog/datasets'
    cfg.data.name = 'AnankeDR3_m12i_lsr0'
    cfg.data.num_datasets = 10
    cfg.data.features = ['ra', 'dec', 'pmra', 'pmdec']

    # logging configuration
    cfg.workdir = '/ocean/projects/phy210068p/tvnguyen/accreted_catalog/logging'
    cfg.enable_progress_bar = False
    cfg.overwrite = False

    # training configuration
    # batching and shuffling
    cfg.train_frac = 0.9
    cfg.train_batch_size = 1024
    cfg.num_workers = 1

    # evaluation configuration
    cfg.eval_batch_size = 1024

    # loss configuration
    cfg.class_weights = None

    # model configuration
    cfg.model = config_dict.ConfigDict()
    cfg.model.input_dim = len(cfg.data.features)
    cfg.model.output_dim = 2  # also the number of classe
    cfg.model.hidden_sizes = [100, 50]
    cfg.model.activation = config_dict.ConfigDict()
    cfg.model.activation.name = 'ReLU'

    # optimizer and scheduler configuration
    cfg.optimizer = config_dict.ConfigDict()
    cfg.optimizer.name = 'Adam'
    cfg.optimizer.lr = 1e-3
    cfg.optimizer.betas = (0.9, 0.999)
    cfg.optimizer.weight_decay = 0
    cfg.optimizer.eps = 1e-8
    cfg.scheduler = config_dict.ConfigDict()
    cfg.scheduler.name = None
    # cfg.scheduler.decay_steps = 100_000  # include warmup steps
    # cfg.scheduler.warmup_steps = 5000
    # cfg.scheduler.eta_min = 1e-6
    # cfg.scheduler.interval = 'step'

    # training loop configuration
    cfg.num_epochs = 100
    cfg.patience = 100
    cfg.monitor = 'val_loss'
    cfg.mode = 'min'
    cfg.grad_clip = 0.5
    cfg.save_top_k = 5
    cfg.accelerator = 'gpu'

    return cfg
