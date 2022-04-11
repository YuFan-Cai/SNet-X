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
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs
from scipy.ndimage.filters import gaussian_filter, median_filter
import random
import cv2
import copy


rotate = [0, 90, 180, 270, 360, 720]
mode = [0, 1, 2, 3]
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cudnn.benchmark = True
eps = 1e-7
window = [9, 11, 13, 15]
combine = [16, 32, 64, 128, 256]


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

        win = random.choice(window)
        smooth_1 = gaussian_filter(input, sigma=win, truncate=1.0)
        smooth_2 = median_filter(input, size=win)

        inp = np.zeros_like(input, dtype=np.float64)
        lab = np.zeros_like(label, dtype=np.float64)
        search_h = random.choice(combine)
        search_w = random.choice(combine)
        for i in range(height // search_h):
            for j in range(width // search_w):
                M = random.choice(mode)
                start_h = i * search_h
                start_w = j * search_w
                if M == 0:
                    inp[start_h:start_h+search_h, start_w:start_w+search_w] = input[start_h:start_h+search_h, start_w:start_w+search_w]
                    lab[start_h:start_h+search_h, start_w:start_w+search_w] = 0
                elif M == 1:
                    inp[start_h:start_h+search_h, start_w:start_w+search_w] = label[start_h:start_h+search_h, start_w:start_w+search_w]
                    lab[start_h:start_h+search_h, start_w:start_w+search_w] = 1
                elif M == 2:
                    inp[start_h:start_h+search_h, start_w:start_w+search_w] = smooth_1[start_h:start_h+search_h, start_w:start_w+search_w]
                    lab[start_h:start_h+search_h, start_w:start_w+search_w] = 2
                else:
                    inp[start_h:start_h+search_h, start_w:start_w+search_w] = smooth_2[start_h:start_h+search_h, start_w:start_w+search_w]
                    lab[start_h:start_h+search_h, start_w:start_w+search_w] = 2

        register_noise = inp - input
        lab[register_noise == 0] = 0

        register_clean = inp - label
        lab[register_clean == 0] = 1
        return inp, lab


def valid(Model, Mode):
    if Mode == 0:
        Test_file = '../../Data/Test/Virtual-SAR/Test-A.tiff'
        Refer_img = '../../Data/Test/Virtual-SAR/Refer-A.tiff'
        Real_img = '../../Data/Test/Sentinel-1A/Test-VH.tiff'
    else:
        Test_file = '../../Data/Test/Virtual-SAR/Test-B.tiff'
        Refer_img = '../../Data/Test/Virtual-SAR/Refer-B.tiff'
        Real_img = '../../Data/Test/CP-SAR/Test-RR.tiff'

    # Model = copy.deepcopy(Model).to('cpu')

    img = gdal.Open(Test_file)
    width = img.RasterXSize
    height = img.RasterYSize
    input = img.ReadAsArray(0, 0, width, height)

    img = gdal.Open(Refer_img)
    width = img.RasterXSize
    height = img.RasterYSize
    label = img.ReadAsArray(0, 0, width, height)

    img = gdal.Open(Real_img)
    width = img.RasterXSize
    height = img.RasterYSize
    T = img.ReadAsArray(0, 0, width, height)

    Model.eval()
    with torch.no_grad():
        input = torch.tensor(input).to("cuda" if torch.cuda.is_available() else "cpu").float().unsqueeze(dim=0)
        label = torch.tensor(label).to("cuda" if torch.cuda.is_available() else "cpu").float().unsqueeze(dim=0)
        T = torch.tensor(T).to("cuda" if torch.cuda.is_available() else "cpu").float().unsqueeze(dim=0)

        pre, _ = Model(input)
        pre = torch.max(F.softmax(pre, dim=1), dim=1)[1]

        k_1 = torch.zeros_like(pre, dtype=torch.float)
        k_2 = torch.zeros_like(pre, dtype=torch.float)
        k_3 = torch.zeros_like(pre, dtype=torch.float)

        k_1[pre == 0] = 1
        k_2[pre == 1] = 1
        k_3[pre == 2] = 1

        print('Noisy:{a}, Clean:{b}, Smooth:{c}'.format(a=k_1.mean().item(), b=k_2.mean().item(), c=k_3.mean().item()))

        plt.imshow(pre[0, :, :].cpu(), vmin=0, vmax=2)
        plt.colorbar()
        plt.show()
        plt.close()

        key_1 = k_1.mean().item()

        pre, _ = Model(label)
        pre = torch.max(F.softmax(pre, dim=1), dim=1)[1]

        k_1 = torch.zeros_like(pre, dtype=torch.float)
        k_2 = torch.zeros_like(pre, dtype=torch.float)
        k_3 = torch.zeros_like(pre, dtype=torch.float)

        k_1[pre == 0] = 1
        k_2[pre == 1] = 1
        k_3[pre == 2] = 1

        print('Noisy:{a}, Clean:{b}, Smooth:{c}'.format(a=k_1.mean().item(), b=k_2.mean().item(), c=k_3.mean().item()))

        plt.imshow(pre[0, :, :].cpu(), vmin=0, vmax=2)
        plt.colorbar()
        plt.show()
        plt.close()

        key_2 = k_2.mean().item()

        pre, _ = Model(T)
        pre = torch.max(F.softmax(pre, dim=1), dim=1)[1]

        k_1 = torch.zeros_like(pre, dtype=torch.float)
        k_2 = torch.zeros_like(pre, dtype=torch.float)
        k_3 = torch.zeros_like(pre, dtype=torch.float)

        k_1[pre == 0] = 1
        k_2[pre == 1] = 1
        k_3[pre == 2] = 1

        print('Noisy:{a}, Clean:{b}, Smooth:{c}'.format(a=k_1.mean().item(), b=k_2.mean().item(), c=k_3.mean().item()))

        plt.imshow(pre[0, :, :].cpu(), vmin=0, vmax=2)
        plt.colorbar()
        plt.show()
        plt.close()

        key_3 = k_1.mean().item()

        score = (key_1 + key_2 + key_3) / 3
        print('Score:', score)
        return score


def train(train_data, Model, op, sch, save_path, Ep, Mode):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    Best = valid(Model, Mode)
    ctr = nn.CrossEntropyLoss()
    for i in range(Ep):
        with tqdm(total=len(train_data)) as t:
            t.set_description('Epoch %s' % i)
            Model.train()
            step = 1
            show_loss = 0.0
            for img, label in train_data:
                img = img.to("cuda" if torch.cuda.is_available() else "cpu").float()
                label = label.to("cuda" if torch.cuda.is_available() else "cpu").long()

                pre_A, pre_B = Model(img)
                res = torch.mean(torch.square(pre_A - pre_B))
                loss = ctr(pre_A, label) + res

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
        label_path = '../../Data/Train/Clean_A/'
        mode_name = 'model_A.pth'
    else:
        data_path = '../../Data/Train/Noisy_B/'
        label_path = '../../Data/Train/Clean_B/'
        mode_name = 'model_B.pth'

    file_list = os.listdir(data_path)
    DATA = []
    LABEL = []
    batch_size = 16
    for I in range(len(file_list)):
        data_file = data_path + file_list[I]
        DATA.append(data_file)

        label_file = label_path + file_list[I]
        LABEL.append(label_file)

    train_dataset = Data(DATA, LABEL)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)

    Model_saved = './Weight/'
    reset = True
    MODEL = model.Mixer().to("cuda" if torch.cuda.is_available() else "cpu")
    if not reset:
        state_dict = torch.load(Model_saved + mode_name)
        model_dict = MODEL.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        MODEL.load_state_dict(model_dict)
        print(MODEL)

    optimize = optim.Adam(MODEL.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    MODEL, optimize = amp.initialize(MODEL, optimize, opt_level='O1')
    scheduler = lrs.StepLR(optimize, step_size=5, gamma=0.5)
    epoch = 20
    train(train_loader, MODEL, optimize, scheduler, Model_saved, epoch, M0)