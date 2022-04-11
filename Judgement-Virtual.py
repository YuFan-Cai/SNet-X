import numpy as np
from osgeo import gdal
import cv2
import matplotlib.pyplot as plt
import copy


eps = 1e-12


def read(file, coordinate):
    dataset = gdal.Open(file)
    x_, y_ = coordinate
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    data = dataset.ReadAsArray(x_, y_, width, height)
    del dataset
    return data


def ENL(img):
    mean = np.mean(img) ** 2
    std = np.std(img) ** 2
    if std == 0:
        std = eps
    enl = mean / std
    return enl


def SSIM(processed, original, L):
    print('Data-range', L)
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(processed, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(original, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(processed ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(original ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(processed * original, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def PSNR(processed, original, Range):
    MSE = np.mean((processed - original) ** 2)
    return 10 * np.log10(Range ** 2 / MSE)


def Judgement(pro, ref, Range=255):
    F = ENL(pro)
    I = ENL(ref)
    k = np.abs(F - I)
    print('ENL', k)

    k = PSNR(pro, ref, Range)
    print('PSNR', k)

    k = SSIM(pro, ref, Range)
    print('SSIM', k)

    plt.title('Processed')
    plt.imshow(pro)
    plt.colorbar()
    plt.show()
    plt.close()

    plt.title('Reference')
    plt.imshow(ref)
    plt.colorbar()
    plt.show()
    plt.close()


if __name__ == '__main__':
    M0 = 0
    if M0 == 0:
        Refer_img = './Data/Test/Virtual-SAR/Refer-A.tiff'
        Raw_img = './Data/Test/Virtual-SAR/Test-A.tiff'
    else:
        Refer_img = './Data/Test/Virtual-SAR/Refer-B.tiff'
        Raw_img = './Data/Test/Virtual-SAR/Test-B.tiff'

    Process_img = './Clean.tiff'
    Process = read(Process_img, (0, 0))
    Process = np.float64(Process)

    Refer = read(Refer_img, (0, 0))
    Refer = np.float64(Refer)[0:256, 0:256]

    Judgement(Process, Refer, 255)

    Raw = read(Raw_img, (0, 0))
    Raw = np.float64(Raw)[0:256, 0:256]

    R = copy.deepcopy(Refer)
    R[Refer == 0] = 1
    N = Raw / R

    P = copy.deepcopy(Process)
    P[Process == 0] = 1
    Rate_map = Raw / P

    Judgement(Rate_map, N, 2)

    Restore = N * Process

    Judgement(Restore, Raw, 255)