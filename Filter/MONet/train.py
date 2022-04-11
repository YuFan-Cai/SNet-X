from osgeo import gdal
import torch.utils.data as data
import torch.optim as optim
import os
from tqdm import *
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from apex import amp
import model
import torch.optim.lr_scheduler as lrs
import cv2
import random


rotate = [0, 90, 180, 270, 360, 720]  # No mention
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cudnn.benchmark = True
size = 32  # 64×64
eps = 1e-7


def net_scope(net):
    blk = 0
    for l in net.named_children():
        if len(l[1].weight.shape) != 4:
            continue
        blk += np.floor(l[1].weight.shape[-1] / 2)
    return int(blk)


def preparation(I_, blk):
    I_in = np.where(np.equal(I_, 0), eps, I_)
    I_in = np.pad(I_in, ((blk, blk), (blk, blk)), mode='edge')
    return I_in


class Data(data.Dataset):
    def __init__(self, Data_path, Label_path, blk):
        super(Data, self).__init__()
        self.image = Data_path
        self.label = Label_path
        self.blk = blk

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

        h = random.randint(self.blk, height - size - self.blk)
        w = random.randint(self.blk, width - size - self.blk)
        input = input[h - self.blk:h + size + self.blk, w - self.blk:w + size + self.blk]
        label = label[h:h + size, w:w + size]
        label = np.where(np.equal(label, 0), eps, label)
        return input, label


def valid(model_, blk, Mode):
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
    input = preparation(input, blk)

    img = gdal.Open(Refer_img)
    width = img.RasterXSize
    height = img.RasterYSize
    label = img.ReadAsArray(0, 0, width, height)

    model_.eval()
    with torch.no_grad():
        input = torch.tensor(input).to("cuda" if torch.cuda.is_available() else "cpu").float().unsqueeze(dim=0)

        pre = model_(input)

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


def gradient(x, h_x=None, w_x=None):
    if h_x is None and w_x is None:
        h_x = x.size()[1]
        w_x = x.size()[2]
    r = F.pad(x, (0, 1, 0, 0))[:, :, 1:]
    l = F.pad(x, (1, 0, 0, 0))[:, :, :w_x]
    t = F.pad(x, (0, 0, 1, 0))[:, :h_x, :]
    b = F.pad(x, (0, 0, 0, 1))[:, 1:, :]
    grad = torch.pow(torch.pow((r - l) * 0.5, 2) + torch.pow((t - b) * 0.5, 2), 0.5)
    return grad


def train(train_data, model_, op, sch, save_path, Ep, blk, Mode):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    Best = valid(model_, blk, Mode)
    for i in range(Ep):
        with tqdm(total=len(train_data)) as t:
            t.set_description('Epoch %s' % i)
            model_.train()
            step = 1
            show_loss = 0.0
            show_1 = 0.0
            show_2 = 0.0
            show_3 = 0.0
            for img, label in train_data:
                img = img.to("cuda" if torch.cuda.is_available() else "cpu").float()
                label = label.to("cuda" if torch.cuda.is_available() else "cpu").float()

                clean = model_(img)
                clean[clean == 0] += eps

                with torch.no_grad():
                    noise_refer = img[:, blk:-blk, blk:-blk] / label
                    noise_refer = torch.clamp(noise_refer, eps, float('inf'))

                    pre_noise = img[:, blk:-blk, blk:-blk] / clean
                    pre_noise = torch.clamp(pre_noise, eps, float('inf'))

                    gradient_i = gradient(label)
                    gradient_p = gradient(clean)

                loss_1 = torch.mean(torch.square(clean - label))

                if i == 0:
                    afa = 0.1
                elif i == 1:
                    afa = 1e+3
                else:
                    afa = 1e+4
                loss_2 = afa * torch.abs(torch.mean(pre_noise * (torch.log2(pre_noise) - torch.log2(noise_refer))))

                beta = 1
                loss_3 = beta * torch.mean(torch.square(gradient_p - gradient_i))

                del img
                loss = loss_1 + loss_2 + loss_3

                op.zero_grad()
                with amp.scale_loss(loss, op) as scaled_loss:
                    scaled_loss.backward()
                op.step()

                show_loss += loss.item()
                show_1 += loss_1.item()
                show_2 += loss_2.item()
                show_3 += loss_3.item()
                t.set_postfix(loss=show_loss / step, loss1=show_1 / step, loss2=show_2 / step, loss3=show_3 / step)
                step += 1
                t.update(1)
            t.close()

            now = valid(model_, blk, Mode)
            if Mode == 0:
                name = 'model_A'
            else:
                name = 'model_B'
            if now >= Best:
                Best = now
                torch.save(model_.state_dict(), '{}{}.pth'.format(save_path, name))
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

    Model_saved = './Weight/'
    reset = True
    MODEL = model.MONet().to("cuda" if torch.cuda.is_available() else "cpu")
    if not reset:
        state_dict = torch.load(Model_saved + mode_name)
        model_dict = MODEL.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        MODEL.load_state_dict(model_dict)
        print(MODEL)

    BLK = net_scope(MODEL)
    train_dataset = Data(DATA, LABEL, BLK)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)

    optimize = optim.Adam(MODEL.parameters(), lr=1e-4, betas=(0.9, 0.99))
    MODEL, optimize = amp.initialize(MODEL, optimize, opt_level='O1')
    scheduler = lrs.MultiStepLR(optimize, milestones=[30], gamma=0.1)  # [87]
    epoch = 50  # 122
    train(train_loader, MODEL, optimize, scheduler, Model_saved, epoch, BLK, M0)