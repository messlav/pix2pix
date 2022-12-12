import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class Maps(Dataset):
    def __init__(self, dir_name: str, split='train', img_transforms=None):
        self.dir_name = dir_name
        self.split = split
        self.transforms = img_transforms
        self.list_files = os.listdir(dir_name + '/' + split)
        self.transforms = img_transforms

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        file = self.list_files[index]
        file_path = os.path.join(self.dir_name + '/' + self.split, file)

        img = T.ToTensor()(Image.open(file_path))
        input_img = img[:, :, :600]
        tgt_img = img[:, :, 600:]

        if self.transforms:
            input_img = self.transforms(input_img)
            tgt_img = self.transforms(tgt_img)

        return input_img, tgt_img


def test():
    dataset = Maps('../data/maps', 'train')
    img = next(iter(dataset))
    print(img[0].shape, img[1].shape)
    img0 = T.ToPILImage()(img[0])
    img0.show()
    img1 = T.ToPILImage()(img[1])
    img1.show()


if __name__ == '__main__':
    test()
