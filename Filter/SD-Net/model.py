import torch.nn as nn
import torch


class FCN(nn.Module):
    def __init__(self, in_c):
        super(FCN, self).__init__()
        self.fcn = nn.Sequential(nn.Conv2d(in_c, 32, 5, padding=2),
                                 nn.PReLU(),
                                 nn.Conv2d(32, 32, 3, padding=1),
                                 nn.PReLU(),
                                 nn.Conv2d(32, 32, 3, padding=1),
                                 nn.PReLU(),
                                 nn.Conv2d(32, 32, 3, padding=1),
                                 nn.PReLU(),
                                 nn.Conv2d(32, in_c, 1),
                                 nn.PReLU())

    def forward(self, x):
        out = self.fcn(x)
        return out


class DepthwiseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=(1, 1), dilation=(1, 1), bias=False, BN=False):
        super(DepthwiseConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                                   bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                                   bias=bias)
        self.bn = BN
        if self.bn:
            self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
            self.act = nn.PReLU()

    def forward(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        if self.bn:
            x = self.act(self.bn(x))
        return x


class Inception_Module(nn.Module):
    def __init__(self, In, Out, d=2):
        super(Inception_Module, self).__init__()
        self.head = nn.Sequential(nn.BatchNorm2d(num_features=In),
                                  nn.PReLU())

        self.res_way = nn.Conv2d(in_channels=In, out_channels=Out, kernel_size=1, stride=1, padding=0, dilation=1)
        self.layer_1 = DepthwiseConvBlock(in_channels=In, out_channels=Out, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, BN=False)
        self.layer_2 = DepthwiseConvBlock(in_channels=In, out_channels=Out // 4, kernel_size=3, stride=1, padding=(1, d), dilation=(1, d), bias=False, BN=False)
        self.layer_3 = DepthwiseConvBlock(in_channels=In, out_channels=Out // 4, kernel_size=3, stride=1, padding=(d, 1), dilation=(d, 1), bias=False, BN=False)
        self.layer_4 = DepthwiseConvBlock(in_channels=In, out_channels=Out // 4, kernel_size=3, stride=1, padding=(d, d), dilation=(d, d), bias=False, BN=False)
        self.layer_5 = DepthwiseConvBlock(in_channels=In, out_channels=Out // 4, kernel_size=3, stride=1, padding=(d + 1, d + 1), dilation=(d + 1, d + 1), bias=False, BN=False)
        self.concat = nn.Conv2d(Out * 2, Out, 1, stride=1, padding=0, dilation=1, bias=False)

    def forward(self, x):
        res = self.res_way(x)
        x = self.head(x)
        out_1 = self.layer_1(x)
        out_2 = self.layer_2(x)
        out_3 = self.layer_3(x)
        out_4 = self.layer_4(x)
        out_5 = self.layer_5(x)
        out = torch.cat([out_1, out_2, out_3, out_4, out_5], dim=1)
        out = self.concat(out) + res
        return out


class SDNet(nn.Module):
    def __init__(self, channel=1):
        super(SDNet, self).__init__()
        self.fcn = FCN(channel)
        self.head = nn.Sequential(nn.Conv2d(channel * 2, 64, 5, stride=1, padding=2, dilation=1))

        self.body_1 = nn.Sequential(Inception_Module(64, 64))

        self.body_2 = nn.Sequential(Inception_Module(64, 128))

        self.body_3 = nn.Sequential(Inception_Module(128, 128))

        self.body_4 = nn.Sequential(Inception_Module(128, 256))

        self.body_5 = nn.Sequential(Inception_Module(256, 256))

        self.body_6 = nn.Sequential(Inception_Module(256, 256))

        self.body_7 = nn.Sequential(Inception_Module(256, 512))

        self.body_8 = nn.Sequential(Inception_Module(512, 512))

        self.body_9 = nn.Sequential(Inception_Module(512, 512))

        self.body_10 = nn.Sequential(Inception_Module(512, 512))

        self.body_11 = nn.Sequential(Inception_Module(512, 512))

        self.body_12 = nn.Sequential(Inception_Module(512, 512))

        self.tail = nn.Sequential(nn.Conv2d(512, 1024, 3, padding=1),
                                  nn.Conv2d(1024, channel, 3, padding=1))

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        pre = self.fcn(x)
        x = torch.cat([x, pre], dim=1)
        Head = self.head(x)

        Layer_1 = self.body_1(Head)
        Layer_2 = self.body_2(Layer_1)
        Layer_3 = self.body_3(Layer_2)
        Layer_4 = self.body_4(Layer_3)
        Layer_5 = self.body_5(Layer_4)
        Layer_6 = self.body_6(Layer_5)
        Layer_7 = self.body_7(Layer_6)
        Layer_8 = self.body_8(Layer_7)
        Layer_9 = self.body_9(Layer_8)
        Layer_10 = self.body_10(Layer_9)
        Layer_11 = self.body_11(Layer_10)
        Layer_12 = self.body_12(Layer_11)

        out = self.tail(Layer_12)
        return out[:, 0, :, :]