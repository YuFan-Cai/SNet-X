import os
import torch
from tqdm import tqdm
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from osgeo import gdal
import numpy as np
import model
import matplotlib.pyplot as plt
from apex import amp
import cv2
import random


rotate = [0, 90, 180, 270, 360, 720]  # Only 180
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cudnn.benchmark = True
size = 32
eps = 1e-7


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
        return input[h:h+size, w:w+size], label[h:h+size, w:w+size]


def valid(MODEL, Mode):
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

    MODEL.eval()
    with torch.no_grad():
        input = torch.tensor(input).to("cuda" if torch.cuda.is_available() else "cpu").float().unsqueeze(dim=0)
        input = torch.log(input + eps)

        pre = input - MODEL(input)
        pre = torch.exp(pre)

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


def MENL(processed, original):
    mean2_p = torch.mean(processed) ** 2
    std2_p = torch.std(processed) ** 2
    enl_p = mean2_p / (std2_p + 1e-12)

    mean2_o = torch.mean(original) ** 2
    std2_o = torch.std(original) ** 2
    enl_o = mean2_o / (std2_o + 1e-12)

    enl = enl_p - enl_o
    enl = torch.mean(enl)
    return enl


def Train(train, MODEL, op, sch, path, Ep, Mode):
    if not os.path.exists(path):
        os.makedirs(path)
    Best = valid(MODEL, Mode)

    for i in range(Ep):
        with tqdm(total=len(train)) as t:
            t.set_description('Epoch %s' % i)
            MODEL.train()
            loss = 0.0
            step = 1
            for img, label in train:
                img = img.float().to("cuda" if torch.cuda.is_available() else "cpu")
                img = torch.log(img + eps)

                label = label.float().to("cuda" if torch.cuda.is_available() else "cpu")
                refer = torch.log(label + eps)
                noi = img - refer

                pre_noise = MODEL(img)
                pre = img - pre_noise
                pre = torch.exp(pre)

                part1 = torch.mean(torch.square(pre_noise - noi))

                part2 = torch.exp(-1 * MENL(pre, label))

                total_loss = part1 + 0.005 * part2

                op.zero_grad()
                with amp.scale_loss(total_loss, op) as scaled_loss:
                    scaled_loss.backward()
                op.step()

                loss += total_loss.item()
                t.set_postfix(loss=loss / step)
                step += 1
                t.update(1)
        t.close()

        now = valid(MODEL, Mode)
        if Mode == 0:
            name = 'model_A'
        else:
            name = 'model_B'
        if now >= Best:
            Best = now
            torch.save(MODEL.state_dict(), '{}{}.pth'.format(path, name))
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
    Model = model.SDNet().to("cuda" if torch.cuda.is_available() else "cpu")
    if not reset:
        state_dict = torch.load(Model_saved + mode_name)
        model_dict = Model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        Model.load_state_dict(model_dict)
        print(Model)

    optimize = optim.RMSprop(Model.parameters(), alpha=0.9, eps=1e-8, lr=1e-5)
    Model, optimize = amp.initialize(Model, optimize, opt_level='O1')
    scheduler = lrs.StepLR(optimize, step_size=20, gamma=0.5)
    epoch = 50

    Train(train_loader, Model, optimize, scheduler, Model_saved, epoch, M0)