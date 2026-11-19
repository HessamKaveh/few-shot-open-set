import torch.nn as nn


def conv_block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))


class Conv4(nn.Module):
    """The 4-block CNN of the original Prototypical Networks paper (64 channels, 32x32 -> 2x2 -> 256-d)."""
    dim = 256

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(conv_block(3, 64), conv_block(64, 64), conv_block(64, 64), conv_block(64, 64))

    def forward(self, x):
        return self.net(x).flatten(1)


class Block(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.LeakyReLU(0.1),
            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.LeakyReLU(0.1),
            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o))
        self.skip = nn.Sequential(nn.Conv2d(i, o, 1, bias=False), nn.BatchNorm2d(o))
        self.out = nn.Sequential(nn.LeakyReLU(0.1), nn.MaxPool2d(2))

    def forward(self, x):
        return self.out(self.body(x) + self.skip(x))


class ResNet12(nn.Module):
    """ResNet-12 (64-160-320-640), the standard few-shot backbone; 32x32 -> 2x2 -> global average pool -> 640-d."""
    dim = 640

    def __init__(self, num_classes=None):
        super().__init__()
        self.net = nn.Sequential(Block(3, 64), Block(64, 160), Block(160, 320), Block(320, 640),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Linear(640, num_classes) if num_classes else None

    def forward(self, x):
        return self.net(x)
