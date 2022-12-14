import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from configs.dataset_flags_config import DatasetFlagsConfig


class Flags(Dataset):
    def __init__(self, dir_name: str, img_transforms=None):
        self.dir_name = dir_name
        self.transforms = img_transforms
        self.list_files = os.listdir(dir_name)
        self.transforms = img_transforms
        if self.transforms is None:
            self.transforms = T.ToTensor()

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        file = self.list_files[index]
        file_path = os.path.join(self.dir_name, file)

        img = Image.open(file_path)
        img_bw = img.convert('1')

        img = T.ToTensor()(img)
        if img.shape[0] != 3:
            print('delete this file please', file_path)
        img_bw = T.ToTensor()(img_bw)

        img_bw = img_bw.repeat(3, 1, 1)
        if self.transforms:
            img, img_bw = self.transforms(torch.stack([img, img_bw]))

        return img, img_bw


def test():
    dataset = Flags('../data/flags/rgb', DatasetFlagsConfig.train_transforms)
    img = next(iter(dataset))
    print(img[0].shape, img[1].shape)
    img0 = T.ToPILImage()(img[0] * 0.5 + 0.5)
    img0.show()
    img1 = T.ToPILImage()(img[1] * 0.5 + 0.5)
    img1.show()


def test2():
    dataset = Flags('../data/flags/rgb', DatasetFlagsConfig.train_transforms)
    for i, img in enumerate(iter(dataset)):
        pass
    print('all')


if __name__ == '__main__':
    test2()
