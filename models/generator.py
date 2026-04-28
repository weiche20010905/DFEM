"""DFME-style generator: noise z -> 3x32x32 synthetic image."""
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, nz=256, ngf=64, img_size=32, nc=3):
        super().__init__()
        self.init_size = img_size // 4
        self.l1 = nn.Linear(nz, ngf * 2 * self.init_size * self.init_size)
        self.ngf = ngf
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(nc, affine=False),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.size(0), self.ngf * 2, self.init_size, self.init_size)
        return self.conv_blocks(out)
