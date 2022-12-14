from argparse import ArgumentParser
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import DataLoader

from datasets.facades import FacadesDataset
from datasets.maps import Maps
from datasets.flags import Flags
from configs.dataset_facades_config import DatasetFacadesConfig
from configs.dataset_maps_config import DatasetMapsConfig
from configs.dataset_flags_config import DatasetFlagsConfig
from model.generator import Generator


def calc_fid(G: Generator, dataset: str = 'facades', device: str = 'cuda:0'):
    if dataset == 'facades':
        test_dataset = FacadesDataset('data/facades', 'test', DatasetFacadesConfig.test_transforms)
    elif dataset == 'maps':
        test_dataset = Maps('data/maps', 'val', DatasetMapsConfig.test_transforms)
    elif dataset == 'flags':
        test_dataset = Flags('data/flags', 'val', DatasetFlagsConfig.test_transforms)
    else:
        raise NotImplementedError
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             num_workers=8, pin_memory=True)
    fid = FrechetInceptionDistance()
    fid.to(device)
    for i, (tgt_imgs, segm_imgs) in enumerate(test_loader):
        tgt_imgs, segm_imgs = tgt_imgs.to(device), segm_imgs.to(device)
        with torch.no_grad():
            fakes = G(segm_imgs)
        fakes = fakes * 0.5 + 0.5
        tgt_imgs = tgt_imgs * 0.5 + 0.5
        int_real_images = tgt_imgs.mul(255).add(0.5).clamp(0, 255).type(torch.uint8)
        int_generated_images = fakes.mul(255).add(0.5).clamp(0, 255).type(torch.uint8)
        fid.update(int_real_images, real=True)
        fid.update(int_generated_images, real=False)
        print(fid.compute().item())

    return fid.compute().item()


def main(G_path: str, dataset: str):
    nc = 3
    if dataset == 'flags_1d':
        nc = 1
    G = Generator(nc)
    G.load_state_dict(torch.load(G_path)['Generator'])
    device = 'cuda:0'
    G.to(device)
    fid = calc_fid(G, dataset, device)
    print('final FID =', fid)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--G_path", type=str, help="path to model weights")
    parser.add_argument(
        "--dataset", type=str, default='facades', help="dataset. 'maps' or 'facades'")
    args = parser.parse_args()
    main(args.G_path, args.dataset)
