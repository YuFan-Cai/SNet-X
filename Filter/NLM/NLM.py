from osgeo import gdal
import time
import numpy as np


Search = 10
Similarity = 3
eps = 1e-7

Test_box = ['../../Data/Test/Virtual-SAR/Test-A.tiff',
            '../../Data/Test/Virtual-SAR/Test-B.tiff',
            '../../Data/Test/Sentinel-1A/Test-VH.tiff',
            '../../Data/Test/CP-SAR/Test-RR.tiff',
            '../../Data/Test/ERS-1/Test-VV.tiff']

Save_box = ['./NLM-Virtual-SAR-A.tiff',
            './NLM-Virtual-SAR-B.tiff',
            './NLM-Sentinel-1A.tiff',
            './NLM-CP-SAR.tiff',
            './NLM-ERS-1.tiff']


def read(file, coordinate):
    dataset = gdal.Open(file)
    x_, y_ = coordinate
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    data = dataset.ReadAsArray(x_, y_, width, height)
    del dataset
    return data


def NLM(img):
    h, w = img.shape
    temp = np.zeros(shape=(h, w))
    r = Search
    f = Similarity
    img = np.pad(img, ((r+f, r+f), (r+f, r+f)), 'edge')

    for i in range(h):
        for j in range(w):
            sigma = np.std(img[i: i+2*r+1, j: j+2*r+1])
            Beta = 0.4 * sigma
            if Beta == 0:
                Beta += eps
            C = 0
            Q = 0
            for m in range(2*r+1):
               for n in range(2*r+1):
                   distance = np.mean((img[i+r: i+r+2*f+1, j+r: j+r+2*f+1] - img[i+m: i+m+2*f+1, j+n: j+n+2*f+1]) ** 2)
                   weight = np.exp(-np.max(distance - 2 * sigma ** 2, 0) / Beta ** 2)
                   if m == r and n == r:
                       weight = 1
                   C += weight
                   Q += img[i+m, j+n] * weight
            temp[i, j] = Q / C
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
    clean = NLM(pwr)
    clean = np.clip(clean, a_min=0, a_max=np.inf)
    end = time.time()
    print(end - start)
    save_slc(clean, x_size, y_size, Save_box[Num])