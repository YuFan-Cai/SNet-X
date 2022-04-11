import torch.nn as nn
import math


class MaskedConv(nn.Module):
    def __init__(self, In, Out, Size, mode, name):
        super(MaskedConv, self).__init__()
        u, d, l, r = Size
        kernel = (1 + u + d, 1 + l + r)
        if mode and u + d + l + r == 0:
            if name == 'LU':
                self.conv = nn.Sequential(nn.Conv2d(In, Out, kernel_size=kernel),
                                          nn.ZeroPad2d(padding=(1, 0, 0, 0)))
                self.flag = 1
            elif name == 'RD':
                self.conv = nn.Sequential(nn.Conv2d(In, Out, kernel_size=kernel),
                                          nn.ZeroPad2d(padding=(0, 1, 0, 0)))
                self.flag = 2
            else:
                print('Not support this Conv')
                exit()
        elif mode and u + d + l + r == 2:
            if name == 'LU':
                self.conv = nn.Sequential(nn.ZeroPad2d(padding=(l, r, u, d)),
                                          nn.Conv2d(In, Out, kernel_size=kernel),
                                          nn.ZeroPad2d(padding=(0, 0, 1, 0)))
                self.flag = 3
            elif name == 'RD':
                self.conv = nn.Sequential(nn.ZeroPad2d(padding=(l, r, u, d)),
                                          nn.Conv2d(In, Out, kernel_size=kernel),
                                          nn.ZeroPad2d(padding=(0, 0, 0, 1)))
                self.flag = 4
            else:
                print('Not support this Conv')
                exit()
        else:
            self.conv = nn.Sequential(nn.ReLU(False),
                                      nn.ZeroPad2d(padding=(l, r, u, d)),
                                      nn.Conv2d(In, Out, kernel_size=kernel))
            self.flag = 0

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.conv(x)
        if self.flag == 1:
            return x[:, :, :, :w]
        elif self.flag == 2:
            return x[:, :, :, 1:]
        elif self.flag == 3:
            return x[:, :, :h, :]
        elif self.flag == 4:
            return x[:, :, 1:, :]
        else:
            return x


class Block(nn.Module):
    def __init__(self, In, Out, Size_h, Size_v, mode=False, name=None):
        super(Block, self).__init__()
        self.scale = math.sqrt(1 / 2.)
        self.flag = mode

        self.V = MaskedConv(In, Out, Size_v, mode, name)
        self.Feed_V = nn.Sequential(nn.ReLU(True),
                                    nn.Conv2d(Out, Out, kernel_size=1))

        self.H = MaskedConv(In, Out, Size_h, mode, name)
        self.Feed_H = nn.Sequential(nn.ReLU(True),
                                    nn.Conv2d(Out, Out, kernel_size=1))

    def forward(self, x_v, x_h):
        v = self.V(x_v)
        feed = self.Feed_V(v)

        h = (self.H(x_h) + feed) * self.scale
        h = self.Feed_H(h)

        if not self.flag:
            h = (h + x_h) * self.scale
        return v, h


class Combine(nn.Module):
    def __init__(self, In, Out):
        super(Combine, self).__init__()
        self.scale = math.sqrt(1 / 2.)
        self.layer = nn.Sequential(nn.ReLU(True),
                                   nn.Conv2d(In, Out, kernel_size=1))

    def forward(self, lu, rd):
        x = (lu + rd) * self.scale
        x = self.layer(x)
        return x


class Res(nn.Module):
    def __init__(self, In):
        super(Res, self).__init__()
        self.scale = math.sqrt(1 / 2.)
        self.layer = nn.Sequential(nn.ReLU(True),
                                   nn.Conv2d(In, In, kernel_size=1),
                                   nn.ReLU(True),
                                   nn.Conv2d(In, In, kernel_size=1))

    def forward(self, x):
        out = (self.layer(x) + x) * self.scale
        return out


class DPNet(nn.Module):
    def __init__(self, channel=1, num_layer=20, num_res=5):
        super(DPNet, self).__init__()
        features = 64
        filter_size = 3
        self.num_layer = num_layer

        First_v = (0, 0, filter_size // 2, filter_size // 2)
        First_h = (0, 0, 0, 0)
        self.layer_in_lu = Block(channel, features, First_h, First_v, True, 'LU')
        self.layer_in_rd = Block(channel, features, First_h, First_v, True, 'RD')
        self.layer_in = Combine(features, features)

        LU_v = (filter_size // 2, 0, filter_size // 2, filter_size // 2)
        LU_h = (0, 0, filter_size // 2, 0)
        RD_v = (0, filter_size // 2, filter_size // 2, filter_size // 2)
        RD_h = (0, 0, 0, filter_size // 2)

        layer_lu = []
        layer_rd = []
        layer = []
        for _ in range(num_layer):
            layer_lu.append(Block(features, features, LU_h, LU_v, False, 'LU'))
            layer_rd.append(Block(features, features, RD_h, RD_v, False, 'RD'))
            layer.append(Combine(features, features))
        self.layer_lu = nn.Sequential(*layer_lu)
        self.layer_rd = nn.Sequential(*layer_rd)
        self.layer = nn.Sequential(*layer)

        self.scale = math.sqrt(1 / 21.)

        res_layer = []
        for _ in range(num_res):
            res_layer.append(Res(features))
        self.res = nn.Sequential(*res_layer)

        self.layer_out = nn.Sequential(nn.ReLU(True),
                                       nn.Conv2d(features, channel * 2, kernel_size=1))

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        v_lu, h_lu = self.layer_in_lu(x, x)
        v_rd, h_rd = self.layer_in_rd(x, x)
        out = self.layer_in(h_lu, h_rd)

        for i in range(self.num_layer):
            v_lu, h_lu = self.layer_lu[i](v_lu, h_lu)
            v_rd, h_rd = self.layer_rd[i](v_rd, h_rd)
            out += self.layer[i](h_lu, h_rd)

        out = self.res(out * self.scale)
        out = self.layer_out(out)
        return out