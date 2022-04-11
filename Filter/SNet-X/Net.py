import torch.nn as nn
import torch
from einops import rearrange


class CONV_A(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=1):
        super(CONV_A, self).__init__()
        pad = kernel_size // 2 * int(dilation)
        self.pad = nn.ReplicationPad2d((pad, pad, pad, pad))
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=0, stride=1)

        self.linear = nn.Linear(in_channel, out_channel, bias=False)
        torch.nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))

        self.kernel = kernel_size * kernel_size
        kernel = torch.empty(in_channel, self.kernel, out_channel)
        self.weight = torch.nn.Parameter(data=kernel, requires_grad=True)
        torch.nn.init.xavier_uniform_(kernel, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        _, _, h, w = x.shape
        out = self.pad(x)
        out = self.unfold(out)
        out = rearrange(out, 'b (c k) n -> b c n k', k=self.kernel)

        Mean = out.mean(dim=-1, keepdim=True)
        Mean_ = rearrange(Mean, 'b c n l -> b n (c l)', l=1)
        Mean_ = self.linear(Mean_)

        out = torch.sum(torch.einsum('b h n k, h k d -> b h n d', out - Mean, self.weight), dim=1) + Mean_
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
        return out


class CONV_B(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=1):
        super(CONV_B, self).__init__()
        pad = kernel_size // 2 * int(dilation)
        self.pad = nn.ReplicationPad2d((pad, pad, pad, pad))
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=0, stride=1)

        self.linear = nn.Linear(in_channel, out_channel, bias=False)
        torch.nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))

        self.kernel = kernel_size * kernel_size
        kernel = torch.empty(in_channel, self.kernel, out_channel)
        self.weight = torch.nn.Parameter(data=kernel, requires_grad=True)
        torch.nn.init.xavier_uniform_(kernel, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        _, _, h, w = x.shape
        out = self.pad(x)
        out = self.unfold(out)
        out = rearrange(out, 'b (c k) n -> b c n k', k=self.kernel)

        center = out[:, :, :, self.kernel // 2].unsqueeze(dim=-1)
        center_ = rearrange(center, 'b c n l -> b n (c l)', l=1)
        center_ = self.linear(center_)

        out = torch.sum(torch.einsum('b h n k, h k d -> b h n d', out - center, self.weight), dim=1) + center_
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
        return out


class Block_A(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=1, act=True):
        super(Block_A, self).__init__()
        self.kernel = CONV_A(in_channel, out_channel, kernel_size, dilation)

        self.key = act
        if act:
            self.act = nn.ReLU(True)

    def forward(self, x):
        out = self.kernel(x)

        if self.key:
            out = self.act(out)
        return out


class Block_B(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=1, act=True):
        super(Block_B, self).__init__()
        self.kernel = CONV_B(in_channel, out_channel, kernel_size, dilation)

        self.key = act
        if act:
            self.act = nn.ReLU(True)

    def forward(self, x):
        out = self.kernel(x)

        if self.key:
            out = self.act(out)
        return out


class SD_A(nn.Module):
    def __init__(self, channels):
        super(SD_A, self).__init__()
        num_of_layers = 15
        Feature = 64
        D = [1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 1, 1, 1]

        layers = [Block_A(in_channel=channels, out_channel=Feature)]
        for i in range(num_of_layers):
            layers.append(Block_A(in_channel=Feature, out_channel=Feature, dilation=D[i]))
        layers.append(Block_A(in_channel=Feature, out_channel=channels, act=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x


class SD_B(nn.Module):
    def __init__(self, channels):
        super(SD_B, self).__init__()
        num_of_layers = 15
        Feature = 64
        D = [1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 1, 1, 1]

        layers = [Block_B(in_channel=channels, out_channel=Feature)]
        for i in range(num_of_layers):
            layers.append(Block_B(in_channel=Feature, out_channel=Feature, dilation=D[i]))
        layers.append(Block_B(in_channel=Feature, out_channel=channels, act=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x


class Mixer(nn.Module):
    def __init__(self):
        super(Mixer, self).__init__()
        self.model_A = SD_A(channels=1)
        self.model_B = SD_B(channels=1)

    def forward(self, x, key):
        x = x.unsqueeze(dim=1)
        clean_A = self.model_A(x)
        if key:
            clean_B = self.model_B(x)
            return clean_A[:, 0, :, :], clean_B[:, 0, :, :]
        else:
            return clean_A[:, 0, :, :]


class Apply(nn.Module):
    def __init__(self):
        super(Apply, self).__init__()
        self.model_A = SD_A(channels=1)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        clean = self.model_A(x)
        return clean[:, 0, :, :]