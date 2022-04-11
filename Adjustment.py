from osgeo import gdal
import cv2
from tqdm import *
import os


def read(file, coordinate):
    dataset = gdal.Open(file)
    x_, y_ = coordinate
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    data = dataset.ReadAsArray(x_, y_, width, height)
    del dataset
    return data


if __name__ == '__main__':
    mode = 1
    if mode == 0:
        Process_img = './Data/Train/Noisy_A/'
        Save_p = './Data/Train/Noisy_B/'
        Reference = './Data/Train/Clean_A/'
        Save_r = './Data/Train/Clean_B/'
        File = os.listdir(Process_img)
        scale = 2
        if not os.path.exists(Save_p):
            os.makedirs(Save_p)
        if not os.path.exists(Save_r):
            os.makedirs(Save_r)
        with tqdm(total=len(File)) as t0:
            for i in File:
                pro = read(Process_img + i, (0, 0))
                refer = read(Reference + i, (0, 0))
                h, w = pro.shape
                pro = cv2.resize(pro, (h * scale, w * scale), interpolation=cv2.INTER_CUBIC)
                refer = cv2.resize(refer, (h * scale, w * scale), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(Save_p + i, pro[h:2*h, w:2*w])
                cv2.imwrite(Save_r + i, refer[h:2*h, w:2*w])
                t0.update(1)
            t0.close()
    else:
        scale = 2
        test_input = './Data/Test/Virtual-SAR/Test-A.tiff'
        test_label = './Data/Test/Virtual-SAR/Refer-A.tiff'
        pro = read(test_input, (0, 0))
        h, w = pro.shape
        refer = read(test_label, (0, 0))

        pro = cv2.resize(pro, (h * scale, w * scale), interpolation=cv2.INTER_CUBIC)
        pro = pro[(h * scale)//2 - h//2:(h * scale)//2 + h//2, (w * scale)//2 - w//2:(w * scale)//2 + w//2]
        refer = cv2.resize(refer, (h * scale, w * scale), interpolation=cv2.INTER_NEAREST)
        refer = refer[(h * scale)//2 - h//2:(h * scale)//2 + h//2, (w * scale)//2 - w//2:(w * scale)//2 + w//2]

        H, W = pro.shape
        driver = gdal.GetDriverByName('GTiff')
        datatype = gdal.GDT_Float64
        ori = driver.Create('./Data/Test/Virtual-SAR/Test-B.tiff', W, H, 1, datatype)
        ori.GetRasterBand(1).WriteArray(pro)

        ori = driver.Create('./Data/Test/Virtual-SAR/Refer-B.tiff', W, H, 1, datatype)
        ori.GetRasterBand(1).WriteArray(refer)
    exit()