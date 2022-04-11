import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
from train import clip, L, eps
import model
from osgeo import gdal
import matplotlib.pyplot as plt
import time


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
cudnn.benchmark = True

Test_box = ['../../Data/Test/Virtual-SAR/Test-A.tiff',
            '../../Data/Test/Virtual-SAR/Test-B.tiff',
            '../../Data/Test/Sentinel-1A/Test-VH.tiff',
            '../../Data/Test/CP-SAR/Test-RR.tiff',
            '../../Data/Test/ERS-1/Test-VV.tiff']

Save_box = ['./S2V-Virtual-SAR-A.tiff',
            './S2V-Virtual-SAR-B.tiff',
            './S2V-Sentinel-1A.tiff',
            './S2V-CP-SAR.tiff',
            './S2V-ERS-1.tiff']


if __name__ == '__main__':
    M0 = 1
    if M0 == 0:
        model_name = 'model_A.pth'
    else:
        model_name = 'model_B.pth'

    Num = 4
    Test_file = Test_box[Num]
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
    norm = Max - Min
    if Max == Min:
        norm = 1
    input = (input - Min) / norm

    Model_saved = './Weight/'
    Model = model.Speckle2Void().to("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(Model_saved + model_name, map_location=torch.device('cpu'))
    model_dict = Model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    Model.load_state_dict(model_dict)

    Model.eval()
    with torch.no_grad():
        input = torch.tensor(input).to("cuda" if torch.cuda.is_available() else "cpu").float().unsqueeze(dim=0)

        start = time.time()
        out = Model(input, 1)
        afa = out[:, 0, :, :] + 1
        beta = out[:, 1, :, :]
        looks = torch.ones_like(input) * L
        pre = (beta + looks * input) / (looks + afa - 1 + eps)
        pre = pre * (Max - Min) + Min
        pre = torch.clamp(pre, 0, clip)
        pre = torch.clamp(pre, min=0, max=np.inf)
        pre = np.array(pre[0, :, :].cpu(), dtype=np.float64)
        pre[mask == 0] = refer[mask == 0]
        end = time.time()
        print(end - start)

        plt.imshow(pre)
        plt.colorbar()
        plt.show()
        plt.close()

        driver = gdal.GetDriverByName('GTiff')
        datatype = gdal.GDT_Float64
        pre[np.isnan(pre)] = 1

        ori = driver.Create(Save_box[Num], width, height, 1, datatype)
        ori.GetRasterBand(1).WriteArray(pre)