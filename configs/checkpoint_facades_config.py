from dataclasses import dataclass
import torch


@dataclass
class CheckpointFacadesConfig:
    wandb_project = 'pix2pix'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    nc_in = 3
    nc_out = 3
    n_workers = 8
    num_epochs = 1000
    save_epochs = 100
    validate_epochs = 10
    save_images = 3
    batch_size = 1
    save_path = 'checkpoint_weights_facades_l1'
    optim = 'Adam'
    learning_rate = 2e-4
    beta1 = 0.5
    beta2 = 0.999
    mean = 0
    std = 0.02
    lambda_l1 = 100
