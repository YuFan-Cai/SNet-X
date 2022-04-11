from osgeo import gdal
import torch.utils.data as data
import torch.optim as optim
import os
from tqdm import *
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import model
import numpy as np
from apex import amp
import torch.nn as nn
import cv2
import random


rotate = [0, 90, 180, 270, 360, 720]  # No mention
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cudnn.benchmark = True
shift_list = [3, 1]
prob = [0.9, 0.1]
L = 1
size = 32  # 64×64
clip = 500000
norm = 255  # 100000
eps = 1e-7  # 1e-19


class Data(data.Dataset):
    def __init__(self, Data_path):
        super(Data, self).__init__()
        self.image = Data_path

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        dataset_img = gdal.Open(self.image[idx])
        width = dataset_img.RasterXSize
        height = dataset_img.RasterYSize
        input = dataset_img.ReadAsArray(0, 0, width, height)

        center = (width // 2, height // 2)
        sita = random.sample(rotate, 1)[0]
        M = cv2.getRotationMatrix2D(center, sita, 1.0)

        input = cv2.warpAffine(input, M, (width, height))

        h = random.randint(0, height - size)
        w = random.randint(0, width - size)
        input = input[h:h + size, w:w + size]

        indexes = np.where(input > clip)
        mask = np.ones_like(input)
        mask[indexes] = 0

        input = np.clip(input, 0, None)
        medians = np.median(input, axis=[0, 1], keepdims=True)
        input = np.where(input > clip, medians, input)

        input /= norm
        return input, mask


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1.0):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    @staticmethod
    def _tensor_size(t):
        return t.size()[1] * t.size()[2]

    def forward(self, x):
        Batch_size = x.size()[0]
        h_x = x.size()[1]
        w_x = x.size()[2]
        count_h = self._tensor_size(x[:, 1:, :])
        count_w = self._tensor_size(x[:, :, 1:])
        h_tv = torch.pow((x[:, 1:, :] - x[:, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, 1:] - x[:, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / Batch_size


def valid(Model, Mode):
    if Mode == 0:
        Test_file = '../../Data/Test/Virtual-SAR/Test-A.tiff'
        Refer_img = '../../Data/Test/Virtual-SAR/Refer-A.tiff'
    else:
        Test_file = '../../Data/Test/Virtual-SAR/Test-B.tiff'
        Refer_img = '../../Data/Test/Virtual-SAR/Refer-B.tiff'

    img = gdal.Open(Test_file)
    width = img.RasterXSize
    height = img.RasterYSize
    refer = img.ReadAsArray(0, 0, width, height)

    input = img.ReadAsArray(0, 0, width, height)
    indexes = np.where(input > clip)
    mask = np.ones_like(input)
    mask[indexes] = 0

    input = np.clip(input, 0, None)
    medians = np.median(input, axis=[0, 1], keepdims=True)
    input = np.where(input > clip, medians, input)

    Max = np.max(input)
    Min = np.min(input)
    norm_ = Max - Min
    if Max == Min:
        norm_ = 1
    input = (input - Min) / norm_

    img = gdal.Open(Refer_img)
    width = img.RasterXSize
    height = img.RasterYSize
    label = img.ReadAsArray(0, 0, width, height)

    Model.eval()
    with torch.no_grad():
        input = torch.tensor(input).to("cuda" if torch.cuda.is_available() else "cpu").float().unsqueeze(dim=0)

        out = Model(input, 1)
        afa = out[:, 0, :, :] + 1
        beta = out[:, 1, :, :]
        looks = torch.ones_like(input) * L

        pre = (beta + looks * input) / (looks + afa - 1 + eps)
        pre = pre * (Max - Min) + Min
        pre = torch.clamp(pre, 0, clip)
        pre = np.array(pre[0, :, :].cpu(), dtype=np.float64)
        pre[mask == 0] = refer[mask == 0]

        plt.imshow(pre)
        plt.colorbar()
        plt.show()
        plt.close()

        label = np.float64(label)
        MSE = np.mean((pre - label) ** 2)
        psnr = 10 * np.log10(255 ** 2 / MSE)
        print('PSNR', psnr)
        return psnr


def train(train_data, Model, op, sch, save_path, Ep, Mode):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    Best = valid(Model, Mode)
    tv = TVLoss(5e-5)
    for i in range(Ep):
        with tqdm(total=len(train_data)) as t:
            t.set_description('Epoch %s' % i)
            Model.train()
            step = 1
            show_loss = 0.0
            for img, mask in train_data:
                img = img.to("cuda" if torch.cuda.is_available() else "cpu").float()
                mask = mask.to("cuda" if torch.cuda.is_available() else "cpu").float()

                shift = np.random.choice(shift_list, p=prob)
                out = Model(img, shift)
                afa = out[:, 0, :, :] + 1
                beta = out[:, 1, :, :]
                looks = torch.ones_like(img) * L

                pre = (beta + looks * img) / (looks + afa - 1 + eps) * norm
                pre = torch.clamp(pre, 0, clip)

                Beta = torch.lgamma(looks) + torch.lgamma(afa) - torch.lgamma(looks + afa)
                Likelihood = - looks * torch.log(looks) - (looks - 1) * torch.log(img + eps) - afa * torch.log(beta + eps) + Beta + (looks + afa) * torch.log(beta + looks * img + eps)
                Likelihood = torch.sum(Likelihood * mask) / torch.sum(mask)
                loss = Likelihood + tv(pre)

                op.zero_grad()
                with amp.scale_loss(loss, op) as scaled_loss:
                    scaled_loss.backward()
                op.step()

                show_loss += loss.item()
                t.set_postfix(loss=show_loss / step)
                step += 1
                t.update(1)
            t.close()

            now = valid(Model, Mode)
            if Mode == 0:
                name = 'model_A'
            else:
                name = 'model_B'
            if now >= Best:
                Best = now
                torch.save(Model.state_dict(), '{}{}.pth'.format(save_path, name))
            if sch is not None:
                sch.step()


if __name__ == '__main__':
    M0 = 0
    if M0 == 0:
        data_path = '../../Data/Train/Noisy_A/'
        mode_name = 'model_A.pth'
    else:
        data_path = '../../Data/Train/Noisy_B/'
        mode_name = 'model_B.pth'

    file_list = os.listdir(data_path)
    DATA = []
    batch_size = 16
    for I in range(len(file_list)):
        data_file = data_path + file_list[I]
        DATA.append(data_file)
    train_dataset = Data(DATA)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)

    Model_saved = './Weight/'
    reset = True
    MODEL = model.Speckle2Void().to("cuda" if torch.cuda.is_available() else "cpu")
    if not reset:
        state_dict = torch.load(Model_saved + mode_name)
        model_dict = MODEL.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        MODEL.load_state_dict(model_dict)
        print(MODEL)

    optimize = optim.Adam(MODEL.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08)
    MODEL, optimize = amp.initialize(MODEL, optimize, opt_level='O1')
    scheduler = None
    epoch = 50
    train(train_loader, MODEL, optimize, scheduler, Model_saved, epoch, M0)