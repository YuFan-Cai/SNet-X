import torch
from pytorch_wavelets import DWTForward, DWTInverse
from osgeo import gdal
import numpy as np
import os
import matplotlib.pyplot as plt
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

Test_box = ['../../Data/Test/Virtual-SAR/Test-A.tiff',
            '../../Data/Test/Virtual-SAR/Test-B.tiff',
            '../../Data/Test/Sentinel-1A/Test-VH.tiff',
            '../../Data/Test/CP-SAR/Test-RR.tiff',
            '../../Data/Test/ERS-1/Test-VV.tiff']

Save_box = ['./POAC-Virtual-SAR-A.tiff',
            './POAC-Virtual-SAR-B.tiff',
            './POAC-Sentinel-1A.tiff',
            './POAC-CP-SAR.tiff',
            './POAC-ERS-1.tiff']


def wavelet_transform(x):
    x = x.unsqueeze(dim=0)
    DWT = DWTForward(J=1, mode='symmetric', wave='db3')
    ll, feature = DWT(x)
    return ll[0, 0, :, :], feature[0][0, 0, 0, :, :], feature[0][0, 0, 1, :, :], feature[0][0, 0, 2, :, :]


def poac(ll, lh, hl, hh):
    slh = torch.trace(torch.matmul(ll, torch.transpose(lh, dim0=0, dim1=1))) / torch.trace(torch.matmul(ll, torch.transpose(ll, dim0=0, dim1=1)))
    shl = torch.trace(torch.matmul(ll, torch.transpose(hl, dim0=0, dim1=1))) / torch.trace(torch.matmul(ll, torch.transpose(ll, dim0=0, dim1=1)))
    shh = torch.trace(torch.matmul(ll, torch.transpose(hh, dim0=0, dim1=1))) / torch.trace(torch.matmul(ll, torch.transpose(ll, dim0=0, dim1=1)))
    Bdash = slh * ll
    Cdash = shl * ll
    Ddash = shh * ll
    return torch.cat((torch.unsqueeze(Bdash, 0), torch.unsqueeze(Cdash, 0), torch.unsqueeze(Ddash, 0)), dim=0)


def wavelet_inverse(ll, feature):
    ll = ll.unsqueeze(dim=0).unsqueeze(dim=0)
    feature = feature.unsqueeze(dim=0).unsqueeze(dim=0)
    IDWT = DWTInverse(wave="db3", mode="symmetric")
    return IDWT((ll, [feature]))


if __name__ == '__main__':
    Num = 4
    Test_file = Test_box[Num]
    img = gdal.Open(Test_file)
    width = img.RasterXSize
    height = img.RasterYSize
    input = img.ReadAsArray(0, 0, width, height)

    with torch.no_grad():
        input = torch.tensor(input).to("cuda" if torch.cuda.is_available() else "cpu").float().unsqueeze(dim=0)

        start = time.time()
        LL, LH, HL, HH = wavelet_transform(input)
        pre = poac(LL, LH, HL, HH)
        pre = wavelet_inverse(LL, pre)
        pre = torch.clamp(pre, min=0, max=np.inf)
        end = time.time()
        print(end - start)

        plt.imshow(pre[0, 0, :, :].cpu())
        plt.colorbar()
        plt.show()
        plt.close()

        driver = gdal.GetDriverByName('GTiff')
        datatype = gdal.GDT_Float64
        new_data = np.array(pre.squeeze(dim=0).squeeze(dim=0).cpu())
        new_data[np.isnan(new_data)] = 1

        ori = driver.Create(Save_box[Num], width, height, 1, datatype)
        ori.GetRasterBand(1).WriteArray(new_data)