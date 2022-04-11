import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                                  nn.BatchNorm2d(ch_out),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                                  nn.BatchNorm2d(ch_out),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2),
                                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                                nn.BatchNorm2d(ch_out),
                                nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                                 nn.BatchNorm2d(F_int))

        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                                 nn.BatchNorm2d(F_int))

        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
                                 nn.BatchNorm2d(1),
                                 nn.Sigmoid())

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttU_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=4):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        return d1


class DRN(nn.Module):
    def __init__(self, channels=1, out_ch=3):
        super(DRN, self).__init__()
        features = 64
        self.layer_1 = nn.Sequential(nn.Conv2d(channels, features, kernel_size=3, stride=1, padding=1, dilation=1),
                                     nn.ReLU(True))
        self.layer_2 = nn.Sequential(nn.Conv2d(features, features, kernel_size=3, stride=1, padding=2, dilation=2),
                                     nn.ReLU(True),
                                     nn.Conv2d(features, features, kernel_size=3, stride=1, padding=3, dilation=3),
                                     nn.ReLU(True))
        self.layer_3 = nn.Sequential(nn.Conv2d(features, features, kernel_size=3, stride=1, padding=4, dilation=4),
                                     nn.ReLU(True))
        self.layer_4 = nn.Sequential(nn.Conv2d(features, features, kernel_size=3, stride=1, padding=3, dilation=3),
                                     nn.ReLU(True),
                                     nn.Conv2d(features, features, kernel_size=3, stride=1, padding=2, dilation=2),
                                     nn.ReLU(True))
        self.layer_5 = nn.Conv2d(features, out_ch, kernel_size=3, stride=1, padding=1, dilation=1)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out) + out
        out = self.layer_3(out)
        out = self.layer_4(out) + out
        out = self.layer_5(out)
        return out


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class conv_block_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block_down, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

    def forward(self, x):
        x = self.conv(x)
        return x


class upsample(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor=2):
        super(upsample, self).__init__()
        self.conv1x1 = conv1x1(in_ch, out_ch)
        self.scale_factor = scale_factor

    def forward(self, H):
        H = self.conv1x1(H)
        H = F.interpolate(H, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        return H


class FCN(nn.Module):
    def __init__(self, in_ch=1, out_ch=3):
        super(FCN, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.block1 = conv_block_down(in_ch, 64)
        self.block2 = conv_block_down(64, 128)
        self.block3 = conv_block_down(128, 256)
        self.block4 = conv_block_down(256, 512)
        self.block5 = conv_block_down(512, 512)
        self.upsample1 = upsample(512, 512, 2)
        self.upsample2 = upsample(512, 256, 2)
        self.upsample3 = upsample(256, out_ch, 8)

    def forward(self, x):
        block1_x = self.block1(x)
        block2_x = self.block2(block1_x)
        block3_x = self.block3(block2_x)
        block4_x = self.block4(block3_x)
        block5_x = self.block5(block4_x)
        upsample1 = self.upsample1(block5_x)
        x = torch.add(upsample1, block4_x)
        upsample2 = self.upsample2(x)
        x = torch.add(upsample2, block3_x)
        x = self.upsample3(x)
        return x


class Mixer(nn.Module):
    def __init__(self):
        super(Mixer, self).__init__()
        N = 3
        self.model_A = AttU_Net(output_ch=N)
        self.model_B = DRN(out_ch=N)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        eval_A = self.model_A(x)
        eval_B = self.model_B(x)
        return eval_A, eval_B


class Apply(nn.Module):
    def __init__(self):
        super(Apply, self).__init__()
        N = 3
        self.model_A = AttU_Net(output_ch=N)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        Eval = self.model_A(x)
        return Eval
