import torch
import torch.nn as nn
# from generator import DownBlock as Block


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4,
                              stride=stride, padding=1, bias=False, padding_mode='reflect')
        self.batch_norm = nn.BatchNorm2d(out_channels)
        # self.batch_norm = nn.InstanceNorm2d(out_channels, affine=True)  # might be better. or not :)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.first_block = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )
        self.block1 = Block(64, 128, stride=2)
        self.block2 = Block(128, 256, stride=2)
        self.block3 = Block(256, 512, stride=1)
        self.final_block = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect')

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                torch.nn.init.normal_(m.weight, std=0.02)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x, y):
        x = self.first_block(torch.cat([x, y], dim=1))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.final_block(x)
        return x


def test():
    x = torch.randn((10, 3, 256, 256))  # b x nc x w x h
    y = torch.randn((10, 3, 256, 256))  # b x nc x w x h
    D = Discriminator(x.shape[1])
    print(D(x, y).shape)


if __name__ == '__main__':
    test()
