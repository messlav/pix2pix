import torch
from torch.nn import init
import torchvision.transforms as T


def show_images(train_dataset, test_dataset):
    img = next(iter(train_dataset))
    print(img[0].shape, img[1].shape)
    print(torch.min(img[0]), torch.max(img[0]))
    img0 = T.ToPILImage()(img[0] * 0.5 + 0.5)
    img0.show()
    img1 = T.ToPILImage()(img[1] * 0.5 + 0.5)
    img1.show()

    img = next(iter(test_dataset))
    print(img[0].shape, img[1].shape)
    img0 = T.ToPILImage()(img[0] * 0.5 + 0.5)
    img0.show()
    img1 = T.ToPILImage()(img[1] * 0.5 + 0.5)
    img1.show()


def init_weights(net, mean, std):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, mean, std)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, mean)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, std)
            init.constant_(m.bias.data, mean)

    net.apply(init_func)
