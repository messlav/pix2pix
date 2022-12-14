import torch
import os
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import torch.nn as nn

from configs.checkpoint_facades_config import CheckpointFacadesConfig
from configs.checkpoint_maps_config import CheckpointMapsConfig
from configs.train_flags_config import TrainFlagsConfig
from configs.dataset_facades_config import DatasetFacadesConfig
from configs.dataset_maps_config import DatasetMapsConfig
from configs.dataset_flags_config import DatasetFlagsConfig
from model.generator import Generator
from model.discriminator import Discriminator
from datasets.facades import FacadesDataset
from datasets.maps import Maps
from datasets.flags import Flags
from loss.l1_loss import l1_loss
from utils.wandb_writer import WanDBWriter
from utils.utils import show_images, init_weights, set_random_seed


def main(dataset: str):
    set_random_seed(3407)
    if dataset == 'facades':
        # configs
        checkpoint_config = CheckpointFacadesConfig()
        dataset_config = DatasetFacadesConfig()
        print('using', checkpoint_config.device)
        # data
        train_transforms = dataset_config.train_transforms
        test_transforms = dataset_config.train_transforms
        train_dataset = FacadesDataset('data/facades', 'train', train_transforms)
        test_dataset = FacadesDataset('data/facades', 'test', test_transforms)
    elif dataset == 'maps':
        # configs
        checkpoint_config = CheckpointMapsConfig()
        dataset_config = DatasetMapsConfig()
        print('using', checkpoint_config.device)
        # data
        train_transforms = dataset_config.train_transforms
        test_transforms = dataset_config.train_transforms
        train_dataset = Maps('data/maps', 'train', train_transforms)
        test_dataset = Maps('data/maps', 'val', test_transforms)
    elif dataset == 'flags':
        # configs
        checkpoint_config = TrainFlagsConfig()
        dataset_config = DatasetFlagsConfig()
        print('using', checkpoint_config.device)
        # data
        train_transforms = dataset_config.train_transforms
        test_transforms = dataset_config.train_transforms
        train_dataset = Flags('data/flags', 'train', train_transforms)
        test_dataset = Flags('data/flags', 'val', test_transforms)
    else:
        raise NotImplementedError
    # show_images(train_dataset, test_dataset)
    train_loader = DataLoader(train_dataset, batch_size=checkpoint_config.batch_size,
                              shuffle=True, num_workers=checkpoint_config.n_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=checkpoint_config.save_images,
                             shuffle=False, num_workers=checkpoint_config.n_workers, pin_memory=True)
    # model
    G = Generator(checkpoint_config.nc)
    # init_weights(G, checkpoint_config.mean, checkpoint_config.std)  # made it automatically
    G = G.to(checkpoint_config.device)

    D = Discriminator(checkpoint_config.nc)
    # init_weights(D, checkpoint_config.mean, checkpoint_config.std)  # made it automatically
    D = D.to(checkpoint_config.device)
    # loss, optimizer and hyperparameters
    current_step = 0
    loss_l1 = nn.L1Loss()
    gan_loss = nn.BCEWithLogitsLoss()  # TODO: make own loss
    optimizer_G = torch.optim.Adam(G.parameters(), lr=checkpoint_config.learning_rate,
                                   betas=(checkpoint_config.beta1, checkpoint_config.beta2))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=checkpoint_config.learning_rate,
                                   betas=(checkpoint_config.beta1, checkpoint_config.beta2))
    scaler_G = torch.cuda.amp.GradScaler()
    scaler_D = torch.cuda.amp.GradScaler()
    os.makedirs(checkpoint_config.save_path, exist_ok=True)
    logger = WanDBWriter(checkpoint_config)
    logger.watch_model(G)
    # train
    G.train()
    D.train()
    tqdm_bar = tqdm(total=checkpoint_config.num_epochs * len(train_loader) - current_step)
    for epoch in range(checkpoint_config.num_epochs):
        for i, (tgt_imgs, segm_imgs) in enumerate(train_loader):
            current_step += 1
            tqdm_bar.update(1)
            logger.set_step(current_step)

            tgt_imgs, segm_imgs = tgt_imgs.to(checkpoint_config.device), segm_imgs.to(checkpoint_config.device)
            # Discriminator
            with torch.cuda.amp.autocast():
                fake = G(segm_imgs)
                net_D_real = D(segm_imgs, tgt_imgs)
                net_D_fake = D(segm_imgs, fake.detach())
                D_real_loss = gan_loss(net_D_real, torch.ones_like(net_D_real))
                D_fake_loss = gan_loss(net_D_fake, torch.zeros_like(net_D_real))
                D_loss = D_real_loss + D_fake_loss

            optimizer_D.zero_grad()
            scaler_D.scale(D_loss).backward()
            scaler_D.step(optimizer_D)
            scaler_D.update()

            # Generator
            with torch.cuda.amp.autocast():
                net_D_fake = D(segm_imgs, fake)
                G_fake_loss = gan_loss(net_D_fake, torch.ones_like(net_D_fake))
                l1 = loss_l1(fake, tgt_imgs) * checkpoint_config.lambda_l1
                G_loss = G_fake_loss + l1

            optimizer_G.zero_grad()
            scaler_G.scale(G_loss).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()

            logger.add_scalar('discriminator_loss', D_loss.item())
            logger.add_scalar('generator_loss', G_loss.item())
            logger.add_scalar('l1_loss', l1.item())
            logger.add_scalar('gan_loss', G_fake_loss.item())
            logger.add_scalar('discriminator_real_loss', D_real_loss.item())
            logger.add_scalar('discriminator_fake_loss', D_fake_loss.item())

        # save G
        if epoch != 0 and epoch % checkpoint_config.save_epochs == 0:
            torch.save({'Generator': G.state_dict(), 'optimizer': optimizer_G.state_dict(
            )}, os.path.join(checkpoint_config.save_path, 'checkpoint_%d.pth.tar' % epoch))

        # validate and wandb log
        if epoch % checkpoint_config.validate_epochs == 0 or epoch == checkpoint_config.num_epochs - 1:
            # add val images
            tgt_imgs, segm_imgs = next(iter(test_loader))
            tgt_imgs, segm_imgs = tgt_imgs.to(checkpoint_config.device), segm_imgs.to(checkpoint_config.device)
            G.eval()
            with torch.no_grad():
                fake = G(segm_imgs)

            for q in range(checkpoint_config.save_images):
                logger.add_image(f'val/segmentation{q}',
                                 segm_imgs[q].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5)
                logger.add_image(f'val/ground_true{q}',
                                 tgt_imgs[q].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5)
                logger.add_image(f'val/prediction{q}',
                                 fake[q].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5)
            # add train images
            tgt_imgs, segm_imgs = next(iter(train_loader))
            tgt_imgs, segm_imgs = tgt_imgs.to(checkpoint_config.device), segm_imgs.to(checkpoint_config.device)
            with torch.no_grad():
                fake = G(segm_imgs)

            for q in range(checkpoint_config.save_images):
                if q >= checkpoint_config.batch_size:
                    break
                logger.add_image(f'train/segmentation{q}',
                                 segm_imgs[q].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5)
                logger.add_image(f'train/ground_true{q}',
                                 tgt_imgs[q].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5)
                logger.add_image(f'train/prediction{q}',
                                 fake[q].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5)
            G.train()

    torch.save({'Generator': G.state_dict(), 'optimizer': optimizer_G.state_dict()},
               os.path.join(checkpoint_config.save_path, 'checkpoint_last.pth.tar'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default='facades', help="dataset. 'maps' or 'facades' or 'flags'")
    args = parser.parse_args()
    main(args.dataset)
