import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class FacadesDataset(Dataset):
    def __init__(self, dir_name: str, split='train', img_transforms=None):
        self.dir_name = dir_name
        self.split = split
        self.transforms = img_transforms
        if self.transforms is None:
            self.transforms = T.ToTensor()

        meta = pd.read_csv(os.path.join(self.dir_name, 'metadata.csv'))
        meta = meta[meta.split == split]

        if split == 'train':
            meta['image_id'] = meta['image_id'].apply(lambda s: s[:-2])

        self.meta = meta.sort_values(by=['domain']).groupby("image_id").agg({'image_path': tuple})

    def __len__(self):
        return self.meta.shape[0]

    def __getitem__(self, index):
        tgt_path, segm_path = self.meta.iloc[index]['image_path']
        tgt_path = os.path.join(self.dir_name, tgt_path.replace('B', 'A'))
        segm_path = os.path.join(self.dir_name, segm_path.replace('A', 'B'))

        tgt_img = self.transforms(Image.open(tgt_path))
        segm_img = self.transforms(Image.open(segm_path))

        return tgt_img, segm_img


def test():
    dataset = FacadesDataset('../data', 'train')
    img = next(iter(dataset))
    print(img[0].shape, img[1].shape)
    img0 = T.ToPILImage()(img[0])
    img0.show()
    img1 = T.ToPILImage()(img[1])
    img1.show()


if __name__ == '__main__':
    test()
