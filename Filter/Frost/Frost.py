import numpy as np
from osgeo import gdal
import time


Half_window_size = 4  # A: 2 / B: 4
a = 3
eps = 1e-7

Test_box = ['../../Data/Test/Virtual-SAR/Test-A.tiff',
            '../../Data/Test/Virtual-SAR/Test-B.tiff',
            '../../Data/Test/Sentinel-1A/Test-VH.tiff',
            '../../Data/Test/CP-SAR/Test-RR.tiff',
            '../../Data/Test/ERS-1/Test-VV.tiff']

Save_box = ['./Frost-Virtual-SAR-A.tiff',
            './Frost-Virtual-SAR-B.tiff',
            './Frost-Sentinel-1A.tiff',
            './Frost-CP-SAR.tiff',
            './Frost-ERS-1.tiff']


def read(file, coordinate):
    dataset = gdal.Open(file)
    x_, y_ = coordinate
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    data = dataset.ReadAsArray(x_, y_, width, height)
    del dataset
    return data


def Frost(img):
    h, w = img.shape
    temp = np.zeros(shape=(h, w))
    N = Half_window_size
    size = 2 * N + 1
    img = np.pad(img, ((N, N), (N, N)), 'edge')
    D = np.zeros(shape=(size, size))

    for k in range(size):
        for l in range(size):
            D[k, l] = np.sqrt((k - N) ** 2 + (l - N) ** 2)

    for i in range(h):
        for j in range(w):
            box = img[i: i+size, j: j+size]
            mul = np.mean(box)
            if mul == 0:
                mul += eps
            var = np.std(box) ** 2
            afa = np.sqrt(np.abs(a * var / mul ** 2))
            W = np.exp(- D * afa) * afa
            temp[i, j] = (box * W).sum() / W.sum()
    return temp


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
    clean = Frost(pwr)
    clean = np.clip(clean, a_min=0, a_max=np.inf)
    end = time.time()
    print(end - start)
    save_slc(clean, x_size, y_size, Save_box[Num])