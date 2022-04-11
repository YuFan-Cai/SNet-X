import torch.nn as nn
import torch


class N2N_L(nn.Module):
    def __init__(self, channel=1):
        super(N2N_L, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(channel, 48, kernel_size=3, padding=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    nn.UpsamplingNearest2d(scale_factor=2))

        self.layer4 = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    nn.UpsamplingNearest2d(scale_factor=2))

        self.layer5 = nn.Sequential(nn.Conv2d(97, 64, kernel_size=3, padding=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    nn.Conv2d(32, channel, kernel_size=3, padding=1, bias=True))

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        out_f = self.layer1(x)
        out = self.layer2(out_f)
        out = self.layer3(out)
        out = torch.cat([out, out_f], dim=1)
        out = self.layer4(out)
        out = torch.cat([out, x], dim=1)
        out = self.layer5(out)
        return out[:, 0, :, :]


class N2N(nn.Module):
    def __init__(self, channel=1):
        super(N2N, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(channel, 48, kernel_size=3, padding=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer4 = nn.Sequential(nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer5 = nn.Sequential(nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer6 = nn.Sequential(nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    nn.UpsamplingNearest2d(scale_factor=2))

        self.layer7 = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    nn.UpsamplingNearest2d(scale_factor=2))

        self.layer8 = nn.Sequential(nn.Conv2d(144, 96, kernel_size=3, padding=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    nn.UpsamplingNearest2d(scale_factor=2))

        self.layer9 = nn.Sequential(nn.Conv2d(144, 96, kernel_size=3, padding=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    nn.UpsamplingNearest2d(scale_factor=2))

        self.layer10 = nn.Sequential(nn.Conv2d(144, 96, kernel_size=3, padding=1, bias=True),
                                     nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                     nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=True),
                                     nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                     nn.UpsamplingNearest2d(scale_factor=2))

        self.layer11 = nn.Sequential(nn.Conv2d(97, 64, kernel_size=3, padding=1, bias=True),
                                     nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                     nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=True),
                                     nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                     nn.Conv2d(32, channel, kernel_size=3, padding=1, bias=True))

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        out_1 = self.layer1(x)
        out_2 = self.layer2(out_1)
        out_3 = self.layer3(out_2)
        out_4 = self.layer4(out_3)
        out = self.layer5(out_4)
        out = self.layer6(out)
        out = torch.cat([out, out_4], dim=1)
        out = self.layer7(out)
        out = torch.cat([out, out_3], dim=1)
        out = self.layer8(out)
        out = torch.cat([out, out_2], dim=1)
        out = self.layer9(out)
        out = torch.cat([out, out_1], dim=1)
        out = self.layer10(out)
        out = torch.cat([out, x], dim=1)
        out = self.layer11(out)
        return out[:, 0, :, :]