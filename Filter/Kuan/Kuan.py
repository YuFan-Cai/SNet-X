from scipy.ndimage.filters import uniform_filter
from osgeo import gdal
import time
import numpy as np


Window_size = 9  # A: 5 / B: 9
Cu = 0.25
Cmax = 0.37
eps = 1e-7

Test_box = ['../../Data/Test/Virtual-SAR/Test-A.tiff',
            '../../Data/Test/Virtual-SAR/Test-B.tiff',
            '../../Data/Test/Sentinel-1A/Test-VH.tiff',
            '../../Data/Test/CP-SAR/Test-RR.tiff',
            '../../Data/Test/ERS-1/Test-VV.tiff']

Save_box = ['./Kuan-Virtual-SAR-A.tiff',
            './Kuan-Virtual-SAR-B.tiff',
            './Kuan-Sentinel-1A.tiff',
            './Kuan-CP-SAR.tiff',
            './Kuan-ERS-1.tiff']


def read(file, coordinate):
    dataset = gdal.Open(file)
    x_, y_ = coordinate
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    data = dataset.ReadAsArray(x_, y_, width, height)
    del dataset
    return data


def Kuan(img):
    size = Window_size

    img_mean = uniform_filter(img, (size, size))

    img_sqr_mean = uniform_filter(img ** 2, (size, size))
    img_variance = img_sqr_mean - img_mean ** 2

    img_mean = np.where(img_mean == 0, eps, img_mean)
    Ci = np.sqrt(img_variance) / img_mean

    T = 1 + Cu ** 2
    Ci = np.where(Ci == 0, eps, Ci)
    img_weights = (1 - Cu ** 2 / Ci ** 2) / T
    img_weights = np.where(Ci <= Cu, 0.0, img_weights)
    img_weights = np.where(Ci >= Cmax, 1.0, img_weights)

    img_output = img_mean + img_weights * (img - img_mean)
    return img_output


def save_slc(new_data, x, y, name):
    driver = gdal.GetDriverByName('GTiff')
    datatype = gdal.GDT_Float64

    ori = driver.Create(name, x, y, 1, datatype)
    ori.GetRasterBand(1).WriteArray(new_data)
    del ori
    del new_data


if __name__ == '__main__':
    Num = 4
    Ori_img = Test_box[Num]
    pwr = read(Ori_img, (0, 0))
    y_size, x_size = pwr.shape

    start = time.time()
    clean = Kuan(pwr)
    clean = np.clip(clean, a_min=0, a_max=np.inf)
    end = time.time()
    print(end - start)
    save_slc(clean, x_size, y_size, Save_box[Num])