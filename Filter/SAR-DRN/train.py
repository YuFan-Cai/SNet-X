from osgeo import gdal
import torch.utils.data as data
import torch.optim as optim
import os
from tqdm import *
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from apex import amp
import torch.optim.lr_scheduler as lrs
import cv2
import random


rotate = [0, 90, 180, 270, 360, 720]  # No mention
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cudnn.benchmark = True
size = 32  # 40×40


class Data(data.Dataset):
    def __init__(self, Data_path, Label_path):
        super(Data, self).__init__()
        self.image = Data_path
        self.label = Label_path

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        dataset_img = gdal.Open(self.image[idx])
        dataset_label = gdal.Open(self.label[idx])
        width = dataset_img.RasterXSize
        height = dataset_img.RasterYSize
        input = dataset_img.ReadAsArray(0, 0, width, height)
        label = dataset_label.ReadAsArray(0, 0, width, height)

        center = (width // 2, height // 2)
        sita = random.sample(rotate, 1)[0]
        M = cv2.getRotationMatrix2D(center, sita, 1.0)

        input = cv2.warpAffine(input, M, (width, height))
        label = cv2.warpAffine(label, M, (width, height))

        h = random.randint(0, height - size)
        w = random.randint(0, width - size)
        return input[h:h + size, w:w + size], label[h:h + size, w:w + size]


class DRN(nn.Module):
    def __init__(self, channels=1):
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
        self.layer_5 = nn.Conv2d(features, channels, kernel_size=3, stride=1, padding=1, dilation=1)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        out = self.layer_1(x)
        out = self.layer_2(out) + out
        out = self.layer_3(out)
        out = self.layer_4(out) + out
        out = self.layer_5(out)
        return out[:, 0, :, :]


def valid(model, Mode):
    if Mode == 0:
        Test_file = '../../Data/Test/Virtual-SAR/Test-A.tiff'
        Refer_img = '../../Data/Test/Virtual-SAR/Refer-A.tiff'
    else:
        Test_file = '../../Data/Test/Virtual-SAR/Test-B.tiff'
        Refer_img = '../../Data/Test/Virtual-SAR/Refer-B.tiff'

    img = gdal.Open(Test_file)
    width = img.RasterXSize
    height = img.RasterYSize
    input = img.ReadAsArray(0, 0, width, height)

    img = gdal.Open(Refer_img)
    width = img.RasterXSize
    height = img.RasterYSize
    label = img.ReadAsArray(0, 0, width, height)

    model.eval()
    with torch.no_grad():
        input = torch.tensor(input).to("cuda" if torch.cuda.is_available() else "cpu").float().unsqueeze(dim=0)

        pre = input - model(input)

        plt.imshow(pre[0, :, :].cpu())
        plt.colorbar()
        plt.show()
        plt.close()

        pre = np.array(pre.cpu(), dtype=np.float64)
        label = np.float64(label)
        MSE = np.mean((pre - label) ** 2)
        psnr = 10 * np.log10(255 ** 2 / MSE)
        print('PSNR', psnr)
        return psnr


def train(train_data, model, op, sch, save_path, Ep, Mode):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    Best = valid(model, Mode)
    for i in range(Ep):
        with tqdm(total=len(train_data)) as t:
            t.set_description('Epoch %s' % i)
            model.train()
            step = 1
            show_loss = 0.0
            for img, label in train_data:
                img = img.to("cuda" if torch.cuda.is_available() else "cpu").float()
                label = label.to("cuda" if torch.cuda.is_available() else "cpu").float()
                refer = img - label

                pre = model(img)

                loss = torch.mean(torch.square(pre - refer))

                op.zero_grad()
                with amp.scale_loss(loss, op) as scaled_loss:
                    scaled_loss.backward()
                op.step()

                show_loss += loss.item()
                t.set_postfix(loss=show_loss / step)
                step += 1
                t.update(1)
            t.close()

            now = valid(model, Mode)
            if Mode == 0:
                name = 'model_A'
            else:
                name = 'model_B'
            if now >= Best:
                Best = now
                torch.save(model.state_dict(), '{}{}.pth'.format(save_path, name))
            if sch is not None:
                sch.step()


if __name__ == '__main__':
    M0 = 0
    if M0 == 0:
        data_path = '../../Data/Train/Noisy_A/'
        label_path = '../../Data/Train/Clean_A/'
        mode_name = 'model_A.pth'
    else:
        data_path = '../../Data/Train/Noisy_B/'
        label_path = '../../Data/Train/Clean_B/'
        mode_name = 'model_B.pth'

    file_list = os.listdir(data_path)
    DATA = []
    LABEL = []
    batch_size = 16  # 128
    for I in range(len(file_list)):
        data_file = data_path + file_list[I]
        DATA.append(data_file)
        label_file = label_path + file_list[I]
        LABEL.append(label_file)
    train_dataset = Data(DATA, LABEL)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)

    Model_saved = './Weight/'
    reset = True
    MODEL = DRN().to("cuda" if torch.cuda.is_available() else "cpu")
    if not reset:
        state_dict = torch.load(Model_saved + mode_name)
        model_dict = MODEL.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        MODEL.load_state_dict(model_dict)
        print(MODEL)

    optimize = optim.Adam(MODEL.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
    MODEL, optimize = amp.initialize(MODEL, optimize, opt_level='O1')
    scheduler = lrs.StepLR(optimize, step_size=10, gamma=0.5)
    epoch = 50
    train(train_loader, MODEL, optimize, scheduler, Model_saved, epoch, M0)