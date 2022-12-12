import torch
import os
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import torch.nn as nn

from configs.checkpoint_facades_config import CheckpointFacadesConfig
from configs.checkpoint_maps_config import CheckpointMapsConfig
from configs.dataset_facades_config import DatasetFacadesConfig
from configs.dataset_maps_config import DatasetMapsConfig
from model.generator import Generator
from datasets.facades import FacadesDataset
from datasets.maps import Maps
from loss.l1_loss import l1_loss
from utils.wandb_writer import WanDBWriter
from utils.utils import show_images, init_weights


def main(dataset: str):
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
    else:
        raise NotImplementedError
    # show_images(train_dataset, test_dataset)
    train_loader = DataLoader(train_dataset, batch_size=checkpoint_config.batch_size,
                              shuffle=True, num_workers=checkpoint_config.n_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=checkpoint_config.save_images,
                             shuffle=False, num_workers=checkpoint_config.n_workers, pin_memory=True)
    # show_images(train_dataset, test_dataset)
    # model
    G = Generator(checkpoint_config.nc)
    init_weights(G, checkpoint_config.mean, checkpoint_config.std)
    G = G.to(checkpoint_config.device)
    # loss, optimizer and hyperparameters
    current_step = 0
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(G.parameters(), lr=checkpoint_config.learning_rate,
                                 betas=(checkpoint_config.beta1, checkpoint_config.beta2))
    os.makedirs(checkpoint_config.save_path, exist_ok=True)
    logger = WanDBWriter(checkpoint_config)
    logger.watch_model(G)
    # scaler = torch.cuda.amp.GradScaler()
    # train
    G.train()
    tqdm_bar = tqdm(total=checkpoint_config.num_epochs * len(train_loader) - current_step)
    for epoch in range(checkpoint_config.num_epochs):
        for i, (tgt_imgs, segm_imgs) in enumerate(train_loader):
            current_step += 1
            tqdm_bar.update(1)
            logger.set_step(current_step)

            tgt_imgs, segm_imgs = tgt_imgs.to(checkpoint_config.device), segm_imgs.to(checkpoint_config.device)
            # with torch.cuda.amp.autocast():
            #     fake = G(segm_imgs)
            fake = G(segm_imgs)

            optimizer.zero_grad()
            loss = loss_fn(fake, tgt_imgs)
            loss.backward()
            optimizer.step()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            logger.add_scalar('loss', loss.item())

        # save model
        if epoch != 0 and epoch % checkpoint_config.save_epochs == 0:
            torch.save({'Generator': G.state_dict(), 'optimizer': optimizer.state_dict(
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
                                 segm_imgs[q].detach().cpu().permute(1, 2, 0).numpy())
                logger.add_image(f'val/ground_true{q}',
                                 tgt_imgs[q].detach().cpu().permute(1, 2, 0).numpy())
                logger.add_image(f'val/prediction{q}',
                                 fake[q].detach().cpu().permute(1, 2, 0).numpy())
            # add train images
            tgt_imgs, segm_imgs = next(iter(train_loader))
            tgt_imgs, segm_imgs = tgt_imgs.to(checkpoint_config.device), segm_imgs.to(checkpoint_config.device)
            with torch.no_grad():
                fake = G(segm_imgs)

            for q in range(checkpoint_config.save_images):
                if q >= checkpoint_config.batch_size:
                    break
                logger.add_image(f'train/segmentation{q}',
                                 segm_imgs[q].detach().cpu().permute(1, 2, 0).numpy())
                logger.add_image(f'train/ground_true{q}',
                                 tgt_imgs[q].detach().cpu().permute(1, 2, 0).numpy())
                logger.add_image(f'train/prediction{q}',
                                 fake[q].detach().cpu().permute(1, 2, 0).numpy())
            G.train()

    torch.save({'Generator': G.state_dict(), 'optimizer': optimizer.state_dict()},
               os.path.join(checkpoint_config.save_path, 'checkpoint_last.pth.tar'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default='facades', help="dataset. 'maps' or 'facades'")
    args = parser.parse_args()
    main(args.dataset)
