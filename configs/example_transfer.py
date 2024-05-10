
from ml_collections import config_dict

def get_config():
    cfg = config_dict.ConfigDict()

    # training seed for reproducibility
    cfg.seed = 45

    # data configuration
    cfg.data = config_dict.ConfigDict()
    cfg.data.root = '/ocean/projects/phy210068p/tvnguyen/accreted_catalog/datasets'
    cfg.data.name = 'GaiaDR3_transfer'
    cfg.data.num_datasets = 10
    cfg.data.features = ['ra', 'dec', 'pmra', 'pmdec', 'parallax', 'radial_velocity']

    # logging configuration
    cfg.workdir = '/ocean/projects/phy210068p/tvnguyen/accreted_catalog/logging'
    cfg.transfer_name = 'small-spine-6'  # name of the pre-trained model to transfer
    cfg.transfer_checkpoint = 'epoch=24-step=1194775.ckpt'  # which checkpoint to start
    cfg.enable_progress_bar = False
    cfg.overwrite = False

    # tranfer configuration
    cfg.transfer_layers = [0, 1]  # which layers to FREEZE, these layers will not be trained
    # whether to use the same class training weights as the training data, keep False
    cfg.transfer_class_weights = False
    # note: class weight is always computed from the fraction between the classes in
    # the training data, the class_weight_scales is used to scale the class weights
    # by an additional factor, e.g. [1., 0.25] will scale the class weights of the
    # second class by 0.25
    cfg.class_weight_scales = [1., 0.25]

    # training configuration
    # batching and shuffling
    cfg.train_frac = 0.9
    cfg.train_batch_size = 1024
    cfg.num_workers = 1

    # evaluation configuration
    cfg.eval_batch_size = 1024

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
