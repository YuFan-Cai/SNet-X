import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import train
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

Save_box = ['./SAR-DRN-Virtual-SAR-A.tiff',
            './SAR-DRN-Virtual-SAR-B.tiff',
            './SAR-DRN-Sentinel-1A.tiff',
            './SAR-DRN-CP-SAR.tiff',
            './SAR-DRN-ERS-1.tiff']


if __name__ == '__main__':
    M0 = 0
    if M0 == 0:
        model_name = 'model_A.pth'
    else:
        model_name = 'model_B.pth'

    Num = 2
    Test_file = Test_box[Num]
    img = gdal.Open(Test_file)
    width = img.RasterXSize
    height = img.RasterYSize
    input = img.ReadAsArray(0, 0, width, height)

    Model_saved = './Weight/'
    Model = train.DRN().to("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(Model_saved + model_name, map_location=torch.device('cpu'))
    model_dict = Model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    Model.load_state_dict(model_dict)

    Model.eval()
    with torch.no_grad():
        input = torch.tensor(input).to("cuda" if torch.cuda.is_available() else "cpu").float().unsqueeze(dim=0)

        start = time.time()
        pre = input - Model(input)
        pre = torch.clamp(pre, min=0, max=np.inf)
        end = time.time()
        print(end - start)

        plt.imshow(pre[0, :, :].cpu())
        plt.colorbar()
        plt.show()
        plt.close()

        driver = gdal.GetDriverByName('GTiff')
        datatype = gdal.GDT_Float64
        new_data = np.array(pre.squeeze(dim=0).cpu())
        new_data[np.isnan(new_data)] = 1

        ori = driver.Create(Save_box[Num], width, height, 1, datatype)
        ori.GetRasterBand(1).WriteArray(new_data)