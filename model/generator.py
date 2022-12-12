import torch
import torch.nn as nn


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4,
                              stride=2, padding=1, bias=False, padding_mode='reflect')
        self.batch_norm = nn.BatchNorm2d(out_channels)
        # self.batch_norm = nn.InstanceNorm2d(out_channels, affine=True)  # might be better. or not :)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=True):
        super(UpBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4,
                                       stride=2, padding=1, bias=False, padding_mode='zeros')
        self.batch_norm = nn.BatchNorm2d(out_channels)
        # self.batch_norm = nn.InstanceNorm2d(out_channels, affine=True)  # might be better. or not :)
        self.dropout = nn.Dropout(0.5) if dropout else None
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.activation(x)
        return x


class Generator(nn.Module):
    def __init__(self, in_channels):
        super(Generator, self).__init__()  # nc x 256 x 256

        self.first_down = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )  # 64 x 128 x 128
        self.down1 = DownBlock(64, 128)  # 128 x 64 x 64
        self.down2 = DownBlock(128, 256)  # 256 x 32 x 32
        self.down3 = DownBlock(256, 512)  # 512 x 16 x 16
        self.down4 = DownBlock(512, 512)  # 512 x 8 x 8
        self.down5 = DownBlock(512, 512)  # 512 x 4 x 4
        self.down6 = DownBlock(512, 512)  # 512 x 2 x 2
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.ReLU()
        )  # 512 x 1 x 1

        self.up1 = UpBlock(512, 512, dropout=True)  # 512 x 2 x 2
        self.up2 = UpBlock(1024, 512, dropout=True)  # 512 x 4 x 4
        self.up3 = UpBlock(1024, 512, dropout=True)  # 512 x 8 x 8
        self.up4 = UpBlock(1024, 512, dropout=False)  # 512 x 8 x 8
        self.up5 = UpBlock(1024, 256, dropout=False)  # 256 x 16 x 16
        self.up6 = UpBlock(512, 128, dropout=False)  # 128 x 32 x 32
        self.up7 = UpBlock(256, 64, dropout=False)  # 64 x 64 x 64
        self.last_up = nn.Sequential(
            nn.ConvTranspose2d(128, in_channels, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            nn.Tanh()
        )  # 3 x 128 x 128

    def forward(self, x):
        # print('x', x.shape)
        d1 = self.first_down(x)
        # print('d1', d1.shape)
        d2 = self.down1(d1)
        # print('d2', d2.shape)
        d3 = self.down2(d2)
        # print('d3', d3.shape)
        d4 = self.down3(d3)
        # print('d4', d4.shape)
        d5 = self.down4(d4)
        # print('d5', d5.shape)
        d6 = self.down5(d5)
        # print('d6', d6.shape)
        d7 = self.down6(d6)
        # print('d7', d7.shape)
        bottleneck = self.bottleneck(d7)
        # print('bottleneck', bottleneck.shape)

        up1 = self.up1(bottleneck)
        # print('up1', up1.shape)
        up2 = self.up2(torch.cat([up1, d7], dim=1))
        # print('up2', up2.shape)
        up3 = self.up3(torch.cat([up2, d6], dim=1))
        # print('up3', up3.shape)
        up4 = self.up4(torch.cat([up3, d5], dim=1))
        # print('up4', up4.shape)
        up5 = self.up5(torch.cat([up4, d4], dim=1))
        # print('up5', up5.shape)
        up6 = self.up6(torch.cat([up5, d3], dim=1))
        # print('up6', up6.shape)
        up7 = self.up7(torch.cat([up6, d2], dim=1))
        # print('up7', up7.shape)
        final = self.last_up(torch.cat([up7, d1], dim=1))
        # print('final', final.shape)
        return final


def test():
    x = torch.randn((10, 3, 256, 256))  # b x nc x w x h
    G = Generator(x.shape[1])
    print(G(x).shape)


if __name__ == '__main__':
    test()
