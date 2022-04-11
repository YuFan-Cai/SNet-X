import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tf


class block(nn.Module):
    def __init__(self, features=64):
        super(block, self).__init__()
        self.mode = nn.Sequential(nn.ZeroPad2d(padding=(0, 0, 1, 0)),
                                  nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(features),
                                  nn.LeakyReLU(True))

    def forward(self, x):
        out = self.mode(x)
        return out[:, :, :-1, :]


class Speckle2Void(nn.Module):
    def __init__(self, channels=1):
        super(Speckle2Void, self).__init__()
        features = 64
        num_of_layers = 15

        self.layer_in_1 = nn.Sequential(nn.ZeroPad2d(padding=(0, 0, 1, 0)),
                                        nn.Conv2d(channels, features, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.LeakyReLU(True))
        layers_1 = []
        for _ in range(num_of_layers):
            layers_1.append(block())
        self.net_1 = nn.Sequential(*layers_1)

        self.layer_out_1 = nn.Sequential(nn.ZeroPad2d(padding=(0, 0, 1, 0)),
                                         nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=False))

        self.layer_in_2 = nn.Sequential(nn.ZeroPad2d(padding=(0, 0, 1, 0)),
                                        nn.Conv2d(channels, features, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.LeakyReLU(True))

        layers_2 = []
        for _ in range(num_of_layers):
            layers_2.append(block())
        self.net_2 = nn.Sequential(*layers_2)

        self.layer_out_2 = nn.Sequential(nn.ZeroPad2d(padding=(0, 0, 1, 0)),
                                         nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=False))

        self.head_1 = nn.Sequential(nn.Conv3d(features, features, kernel_size=(4, 1, 1), stride=1, padding=0, bias=False),
                                    nn.LeakyReLU(True))

        self.head_2 = nn.Sequential(nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0, bias=False),
                                    nn.LeakyReLU(True),
                                    nn.Conv2d(features, 2, kernel_size=1, stride=1, padding=0, bias=False),
                                    nn.ReLU(True))

    @staticmethod
    def dynamic_shift(inp, pad_size):
        u = F.pad(inp, [0, 0, pad_size, 0], mode='constant', value=0)
        u = u[:, :, :-pad_size, :]
        return u

    def forward(self, x, shift):
        x = x.unsqueeze(dim=1)
        out_1 = self.layer_in_1(x)[:, :, :-1, :]
        out_1 = self.net_1(out_1)
        out_1 = self.layer_out_1(out_1)[:, :, :-1, :]
        if shift == 1:
            out_1 = self.dynamic_shift(out_1, 1)
        else:
            out_1 = self.dynamic_shift(out_1, 2)

        out_2 = tf.rotate(x, 180)
        out_2 = self.layer_in_1(out_2)[:, :, :-1, :]
        out_2 = self.net_1(out_2)
        out_2 = self.layer_out_1(out_2)[:, :, :-1, :]
        if shift == 1:
            out_2 = self.dynamic_shift(out_2, 1)
        else:
            out_2 = self.dynamic_shift(out_2, 2)
        out_2 = tf.rotate(out_2, -180)

        out_3 = tf.rotate(x, 90)
        out_3 = self.layer_in_2(out_3)[:, :, :-1, :]
        out_3 = self.net_2(out_3)
        out_3 = self.layer_out_2(out_3)[:, :, :-1, :]
        out_3 = self.dynamic_shift(out_3, 1)
        out_3 = tf.rotate(out_3, -90)

        out_4 = tf.rotate(x, 270)
        out_4 = self.layer_in_2(out_4)[:, :, :-1, :]
        out_4 = self.net_2(out_4)
        out_4 = self.layer_out_2(out_4)[:, :, :-1, :]
        out_4 = self.dynamic_shift(out_4, 1)
        out_4 = tf.rotate(out_4, -270)

        out = torch.stack([out_1, out_2, out_3, out_4], dim=2)
        out = self.head_1(out).squeeze(dim=2)
        out = self.head_2(out)
        return out