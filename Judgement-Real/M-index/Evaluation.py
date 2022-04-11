from osgeo import gdal
import scipy.io
import numpy as np
from scipy.special import gamma
import pywt
import matplotlib.pyplot as plt


eps = 1e-7
L = 1

Test_box = ['../../Data/Test/Sentinel-1A/Test-VH.tiff',
            '../../Data/Test/CP-SAR/Test-RR.tiff',
            '../../Data/Test/ERS-1/Test-VV.tiff']


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
    Std = np.std(img) ** 2
    if Std == 0:
        Std = eps
    Enl = mean / Std
    return Enl


def EPI(processed, original):
    h, w = processed.shape
    p = 0
    o = 0
    for i in range(h-1):
        for j in range(w-1):
            p += np.abs(processed[i, j] - processed[i+1, j]) + np.abs(processed[i, j] - processed[i, j+1])
            o += np.abs(original[i, j] - original[i+1, j]) + np.abs(original[i, j] - original[i, j+1])
    Epi = np.mean(p / o)
    return Epi


def EPD(processed, original, direction):
    h, w = processed.shape
    processed[processed == 0] = 1
    original[original == 0] = 1
    p = 0
    o = 0
    if direction == 0:
        for i in range(h-1):
            for j in range(w-1):
                p += np.abs(processed[i, j] / processed[i, j+1])
                o += np.abs(original[i, j] / original[i, j+1])
    else:
        for i in range(h - 1):
            for j in range(w - 1):
                p += np.abs(processed[i, j] / processed[i+1, j])
                o += np.abs(original[i, j] / original[i+1, j])
    Epd = np.mean(p / o)
    return Epd


def TCR(processed, original):
    max_p = np.max(processed)
    mul_p = np.mean(processed)
    tcr_p = 20 * np.log10(max_p / mul_p)

    max_o = np.max(original)
    mul_o = np.mean(original)
    tcr_o = 20 * np.log10(max_o / mul_o)
    Tcr = np.abs(tcr_p - tcr_o)
    return Tcr


def Qr(processed, original):
    h, w = processed.shape
    R = np.zeros(shape=(h, w))
    _processed = np.zeros(shape=(h, w))
    for i in range(h):
        for j in range(w):
            if processed[i, j] == 0:
                _processed[i, j] += eps
            else:
                _processed[i, j] = processed[i, j]
            R[i, j] = original[i, j] / _processed[i, j]
            if processed[i, j] == 0 and original[i, j] == 0:
                R[i, j] = 1.0

    Ori = Gaussian_Gamma_filter(original)
    mean_O = Ori.mean()
    _R = Gaussian_Gamma_filter(R)
    mean_R = _R.mean()
    Std_O = np.sqrt(np.sum((Ori - mean_O) ** 2))
    Std_O = np.where(Std_O == 0, eps, Std_O)
    Std_R = np.sqrt(np.sum((_R - mean_R) ** 2))
    Std_R = np.where(Std_R == 0, eps, Std_R)
    Afa = np.sum((Ori - mean_O) * (_R - mean_R)) / (Std_O * Std_R)

    sigma_O = Wavelet(original)
    sigma_R = Wavelet(R)

    Lambda = 0.5
    Q = Lambda * np.abs(sigma_O - sigma_R) + (1 - Lambda) * Afa
    return Q


def Gaussian_Gamma_filter(img):
    Afa = 3   # gamma(Afa) = round(out += i ** (2 - 1) * math.exp(-i) for i in range(20))
    Beta = 1.5
    sigma = 4 / np.pi
    h = round(2 * np.pi * sigma)  # 8
    w = round(gamma(Afa) ** 2 * 2 ** (2 * Afa - 1) * Beta / gamma(2 * Afa - 1))  # 8
    P = 8
    U = np.zeros(shape=(h // 2, w))
    L = np.zeros(shape=(h // 2, w))
    H, W = img.shape
    img = np.pad(img, ((h // 4, h // 4), (w // 2, w // 2)), 'edge')
    temp = np.zeros(shape=(P, H, W))

    for p in range(P):
        Theta = np.pi * p / P
        for y in range(0, h // 2):
            for x in range(-w // 2, w // 2):
                _x = x * np.cos(Theta) - y * np.sin(Theta)
                _y = np.abs(x * np.sin(Theta) + y * np.cos(Theta))
                U[y, x] = _y ** (Afa - 1) * np.exp(-(_x ** 2 / 2 / sigma + _y / Beta)) / (np.sqrt(2) * np.pi * sigma * gamma(Afa) * Beta ** Afa)
        U /= np.sum(np.abs(U))

        for y in range(-h // 2, 0):
            for x in range(-w // 2, w // 2):
                _x = x * np.cos(Theta) - y * np.sin(Theta)
                _y = np.abs(x * np.sin(Theta) + y * np.cos(Theta))
                L[y, x] = _y ** (Afa - 1) * np.exp(-(_x ** 2 / 2 / sigma + _y / Beta)) / (np.sqrt(2) * np.pi * sigma * gamma(Afa) * Beta ** Afa)
        L /= np.sum(np.abs(L))

        for i in range(H):
            for j in range(W):
                u = np.sum(img[i: i + h // 2, j: j + w] * U)
                l = np.sum(img[i: i + h // 2, j: j + w] * L)
                if u == 0:
                    u += eps
                if l == 0:
                    l += eps
                m1 = u / l
                m2 = l / u
                temp[p, i, j] = min(m1, m2)
        temp = np.clip(temp, 0, 1)

    MIN = temp.min(axis=0)
    ESM = 1 - MIN
    return ESM


def Wavelet(img):
    cA, (cH, cV, cD) = pywt.dwt2(img, 'haar')
    sigma = 1 / 0.6745 * np.median(np.abs(cD))
    return sigma


if __name__ == '__main__':
    Num = 1
    Test_file = Test_box[Num]
    I = read(Test_file, (0, 0))
    I = np.float64(I)

    Process_img = '../../Clean.tiff'
    F = read(Process_img, (0, 0))
    F = np.float64(F)

    ENL_X = ENL(F)
    print('ENL:', ENL_X)

    tcr = TCR(F, I)
    print('TCR:', tcr)

    epi = EPI(F, I)
    print('EPI:', epi)

    epd = EPD(F, I, 1)
    print('EPD-X:', epd)

    epd = EPD(F, I, 0)
    print('EPD-Y:', epd)

    # q = Qr(F, I)
    # print('Qr:', q)

    record = np.zeros_like(F)
    record[F == 0] = 1
    num = np.sum(record)
    print('Zero points:', num)
    F[F == 0] = 1
    rate_map = I / F

    mul = np.mean(rate_map)
    print('MOR: ', mul)

    var = np.std(rate_map) ** 2
    RI = np.log2(np.abs(mul - 1) + 2) * np.log2(np.abs(var - 1 / L) + 2)
    print('RI: ', RI)

    plt.imshow(rate_map)
    plt.colorbar()
    plt.show()
    plt.close()

    driver = gdal.GetDriverByName('GTiff')
    datatype = gdal.GDT_Float64
    H, W = rate_map.shape
    ori = driver.Create('./Ratio.tiff', W, H, 1, datatype)
    ori.GetRasterBand(1).WriteArray(rate_map)

    save_img = './data/'
    scipy.io.savemat(save_img + 'result.mat', mdict={'I': I, 'F': F})