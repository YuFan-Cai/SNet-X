from scipy.ndimage.filters import uniform_filter
from osgeo import gdal
import time
import numpy as np


Window_size = 5  # A: 5 / B: 9
eps = 1e-7
L = 1

Test_box = ['../../Data/Test/Virtual-SAR/Test-A.tiff',
            '../../Data/Test/Virtual-SAR/Test-B.tiff',
            '../../Data/Test/Sentinel-1A/Test-VH.tiff',
            '../../Data/Test/CP-SAR/Test-RR.tiff',
            '../../Data/Test/ERS-1/Test-VV.tiff']

Save_box = ['./GAMMAP-Virtual-SAR-A.tiff',
            './GAMMAP-Virtual-SAR-B.tiff',
            './GAMMAP-Sentinel-1A.tiff',
            './GAMMAP-CP-SAR.tiff',
            './GAMMAP-ERS-1.tiff']


def read(file, coordinate):
    dataset = gdal.Open(file)
    x_, y_ = coordinate
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    data = dataset.ReadAsArray(x_, y_, width, height)
    del dataset
    return data


def GMAP(img, looks):
    size = Window_size
    Cu = 1 / np.sqrt(looks)
    Cmax = np.sqrt(2) * Cu

    img_mean = uniform_filter(img, (size, size))

    img_sqr_mean = uniform_filter(img ** 2, (size, size))
    img_variance = img_sqr_mean - img_mean ** 2

    img_mean = np.where(img_mean == 0, eps, img_mean)
    Ci = np.sqrt(img_variance) / img_mean

    T = Ci ** 2 - Cu ** 2
    T = np.where(T == 0, eps, T)
    afa = (1 + Cu ** 2) / T
    afa = np.where(afa == 0, eps, afa)

    img_output = ((afa - looks - 1) * img_mean + np.sqrt(np.abs(img_mean ** 2 * (afa - looks - 1) ** 2 + 4 * afa * looks * img))) / (2 * afa)
    img_output = np.where(Ci <= Cu, img_mean, img_output)
    img_output = np.where(Ci > Cmax, img, img_output)
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
    clean = GMAP(pwr, L)
    clean = np.clip(clean, a_min=0, a_max=np.inf)
    end = time.time()
    print(end - start)
    save_slc(clean, x_size, y_size, Save_box[Num])