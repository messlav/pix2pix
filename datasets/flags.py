import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


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
        input_img = self.transforms(img_bw)
        tgt_img = self.transforms(img)
        input_img = input_img.repeat(3, 1, 1)
        return tgt_img, input_img


def test():
    dataset = Flags('../create_flags_dataset/flags/rgb')
    img = next(iter(dataset))
    print(img[0].shape, img[1].shape)
    img0 = T.ToPILImage()(img[0])
    img0.show()
    img1 = T.ToPILImage()(img[1])
    img1.show()


if __name__ == '__main__':
    test()
