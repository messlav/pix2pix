from dataclasses import dataclass
import torchvision.transforms as T


@dataclass
class DatasetFacadesConfig:
    train_transforms = T.Compose([
        T.ToTensor(),
        # T.ColorJitter(),
        T.Resize((286, 286)),
        T.RandomCrop((256, 256)),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        T.RandomHorizontalFlip(p=0.5),
    ])
    test_transforms = T.Compose([
        T.ToTensor(),
        T.Resize((286, 286)),
        T.RandomCrop((256, 256)),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
