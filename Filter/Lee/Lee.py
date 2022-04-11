from scipy.ndimage.filters import uniform_filter
from osgeo import gdal
import time
import numpy as np


Window_size = 9  # A: 5 / B: 9
eps = 1e-7

Test_box = ['../../Data/Test/Virtual-SAR/Test-A.tiff',
            '../../Data/Test/Virtual-SAR/Test-B.tiff',
            '../../Data/Test/Sentinel-1A/Test-VH.tiff',
            '../../Data/Test/CP-SAR/Test-RR.tiff',
            '../../Data/Test/ERS-1/Test-VV.tiff']

Save_box = ['./Lee-Virtual-SAR-A.tiff',
            './Lee-Virtual-SAR-B.tiff',
            './Lee-Sentinel-1A.tiff',
            './Lee-CP-SAR.tiff',
            './Lee-ERS-1.tiff']


def read(file, coordinate):
    dataset = gdal.Open(file)
    x_, y_ = coordinate
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    data = dataset.ReadAsArray(x_, y_, width, height)
    del dataset
    return data


def Lee(img):
    size = Window_size
    mul = 1.0

    img_mean = uniform_filter(img, (size, size))
    clean_mean = img_mean / mul

    img_sqr_mean = uniform_filter(img ** 2, (size, size))
    img_variance = img_sqr_mean - img_mean ** 2

    img_mean = np.where(img_mean == 0, eps, img_mean)
    sigma = np.sqrt(img_variance) / img_mean

    var = (img_variance + img_mean ** 2) / (sigma ** 2 + mul ** 2) - clean_mean ** 2
    T = clean_mean ** 2 * sigma ** 2 + mul ** 2 * var
    T = np.where(T == 0, eps, T)

    img_weights = mul * var / T
    img_output = clean_mean + img_weights * (img - img_mean)
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
    clean = Lee(pwr)
    clean = np.clip(clean, a_min=0, a_max=np.inf)
    end = time.time()
    print(end - start)
    save_slc(clean, x_size, y_size, Save_box[Num])