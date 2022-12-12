from dataclasses import dataclass
import torch


@dataclass
class CheckpointConfig:
    wandb_project = 'pix2pix'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    nc = 3
    n_workers = 4
    num_epochs = 200
    save_epochs = 50
    validate_epochs = 10
    save_images = 3
    batch_size = 1
    save_path = 'checkpoint_weights'
    optim = 'Adam'
    learning_rate = 2e-4
    beta1 = 0.5
    beta2 = 0.999
    mean = 0
    std = 0.02
