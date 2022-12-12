import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

from configs.checkpoint_config import CheckpointConfig
from configs.dataset_config import DatasetConfig
from model.generator import Generator
from datasets.facades import FacadesDataset
from loss.l1_loss import l1_loss
from utils.wandb_writer import WanDBWriter
from utils.utils import show_images, init_weights


def main():
    # configs
    checkpoint_config = CheckpointConfig()
    dataset_config = DatasetConfig()
    print('using', checkpoint_config.device)
    # data
    train_transforms = dataset_config.train_transforms
    test_transforms = dataset_config.train_transforms
    train_dataset = FacadesDataset('data', 'train', train_transforms)
    test_dataset = FacadesDataset('data', 'test', test_transforms)
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
    loss_fn = l1_loss()
    optimizer = torch.optim.Adam(G.parameters(), lr=CheckpointConfig.learning_rate,
                                 betas=(CheckpointConfig.beta1, CheckpointConfig.beta2))
    os.makedirs(CheckpointConfig.save_path, exist_ok=True)
    logger = WanDBWriter(checkpoint_config)
    logger.watch_model(G)
    scaler = torch.cuda.amp.GradScaler()
    # train
    G.train()
    tqdm_bar = tqdm(total=CheckpointConfig.num_epochs * len(train_loader) - current_step)
    for epoch in range(CheckpointConfig.num_epochs):
        for i, (tgt_imgs, segm_imgs) in enumerate(train_loader):
            current_step += 1
            tqdm_bar.update(1)
            logger.set_step(current_step)

            tgt_imgs, segm_imgs = tgt_imgs.to(CheckpointConfig.device), segm_imgs.to(CheckpointConfig.device)
            with torch.cuda.amp.autocast():
                fake = G(segm_imgs)

            optimizer.zero_grad()
            loss = loss_fn(tgt_imgs, fake)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            logger.add_scalar('loss', loss.item())

        # save model
        if epoch != 0 and epoch % CheckpointConfig.save_epochs == 0:
            torch.save({'Generator': G.state_dict(), 'optimizer': optimizer.state_dict(
            )}, os.path.join(CheckpointConfig.save_path, 'checkpoint_%d.pth.tar' % epoch))

        # validate and wandb log
        if epoch % CheckpointConfig.validate_epochs == 0:
            # add val images
            tgt_imgs, segm_imgs = next(iter(test_loader))
            tgt_imgs, segm_imgs = tgt_imgs.to(CheckpointConfig.device), segm_imgs.to(CheckpointConfig.device)
            G.eval()
            with torch.no_grad():
                fake = G(segm_imgs)
                fake = fake * 0.5 + 0.5  # denormalize?

            for q in range(CheckpointConfig.save_images):
                # if q >= CheckpointConfig.batch_size:
                #     break
                logger.add_image(f'val/segmentation{q}', segm_imgs[q].detach().cpu().permute(1, 2, 0).numpy())
                logger.add_image(f'val/ground_true{q}', tgt_imgs[q].detach().cpu().permute(1, 2, 0).numpy())
                logger.add_image(f'val/prediction{q}', fake[q].detach().cpu().permute(1, 2, 0).numpy())
            # add train images
            tgt_imgs, segm_imgs = next(iter(train_loader))
            tgt_imgs, segm_imgs = tgt_imgs.to(CheckpointConfig.device), segm_imgs.to(CheckpointConfig.device)
            with torch.no_grad():
                fake = G(segm_imgs)
                fake = fake * 0.5 + 0.5

            for q in range(CheckpointConfig.save_images):
                if q >= CheckpointConfig.batch_size:
                    break
                logger.add_image(f'train/segmentation{q}', segm_imgs[q].detach().cpu().permute(1, 2, 0).numpy())
                logger.add_image(f'train/ground_true{q}', tgt_imgs[q].detach().cpu().permute(1, 2, 0).numpy())
                logger.add_image(f'train/prediction{q}', fake[q].detach().cpu().permute(1, 2, 0).numpy())
            G.train()

    torch.save({'Generator': G.state_dict(), 'optimizer': optimizer.state_dict()},
               os.path.join(CheckpointConfig.save_path, 'checkpoint_last.pth.tar'))


if __name__ == '__main__':
    main()
