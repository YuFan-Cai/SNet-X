import numpy as np
import pywt
from osgeo import gdal
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from scipy.special import gamma
from scipy.ndimage import filters
from sklearn import svm
import time
import Estimate
import matplotlib.colors as col
import cv2


color = ['purple', 'green', 'red']
cmap = col.ListedColormap(color)
Level = ['a', 'aa', 'aaa', 'aaaa', 'aaaaa', 'aaaaaa', 'aaaaaaa', 'aaaaaaaa']


def dirac(x, sigma=1.5):
    f = (1 / 2 / sigma) * (1 + np.cos(np.pi * x / sigma))
    b = (x <= sigma) & (x >= -sigma)
    return f * b


def heaviside(x, sigma=2):
    return 0.5 * (1 + (2 / 3.1415926) * np.arctan(x / sigma))


def Energy(I, phi, mode, size=5, num=0):
    W = pywt.WaveletPacket2D(data=I, wavelet='haar', maxlevel=8)
    cA = W[Level[num]].data

    m1 = np.mean(cA ** 2)
    m2 = np.mean(cA ** 4)
    m3 = np.mean(cA ** 6)
    m4 = np.mean(cA ** 8)

    # C3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3  # skewness
    C4 = m4 - 3 * m2 ** 2 - 4 * m1 * m3 + 12 * m1 * m2 - 6 * m1 ** 4  # kurtosis
    feature = C4 / np.abs(m2) ** 2

    h1, w1 = I.shape
    N = h1 * w1
    curve = lambda x: (8 * gamma(1 / x) * gamma(9 / x) - 16 * gamma(1 / x) ** 2 * gamma(7 / x) * gamma(3 / x) + 12 * gamma(1 / x) * gamma(5 / x) * gamma(3 / x) ** 2 - 3 * gamma(3 / x) ** 4) / (4 * gamma(1 / x) ** 2 * gamma(5 / x) ** 2 - 4 * gamma(1 / x) * gamma(3 / x) ** 2 * gamma(5 / x) + gamma(3 / x) ** 4)
    # curve = lambda x: (gamma(7 / x) / (gamma(5 / x)) * np.sqrt(gamma(1 / x) / (gamma(5 / x))) + gamma(3 / x) * np.sqrt(N / gamma(1 / x) / gamma(5 / x)) * (2 * N * gamma(3 / x) ** 2 / gamma(1 / x) / gamma(5 / x) - 3))
    List = np.linspace(start=0.1, stop=200, num=int(2e+5))
    # beta = 0.3475
    beta = List[np.argmin(np.abs(curve(List) - feature))]
    afa = np.sqrt(gamma(7 / beta) * m4 / gamma(9 / beta) / m3)
    K = N * beta / afa / gamma(1 / beta)

    kernel = lambda p: K / 2 / np.sqrt(abs(p)) * np.exp(-(np.sqrt(abs(p)) / afa) ** beta)
    k_x = np.linspace(1, 1 + size, size)
    kernel = np.expand_dims(kernel(k_x), axis=1)

    if mode == 0:
        f = convolve(input=heaviside(phi) * I, weights=kernel) / convolve(input=heaviside(phi), weights=kernel)
    else:
        f = convolve(input=(1 - heaviside(phi)) * I, weights=kernel) / convolve(input=1 - heaviside(phi), weights=kernel)
    In = np.square(I - f)
    out = convolve(input=In, weights=kernel)
    return out


def neumann_bound_cond(f):
    g = f.copy()
    g[np.ix_([0, -1], [0, -1])] = g[np.ix_([2, -3], [2, -3])]
    g[np.ix_([0, -1]), 1:-1] = g[np.ix_([2, -3]), 1:-1]
    g[1:-1, np.ix_([0, -1])] = g[1:-1, np.ix_([2, -3])]
    return g


def div(nx, ny):
    [_, nxx] = np.gradient(nx)
    [nyy, _] = np.gradient(ny)
    return nxx + nyy


def forward(img, phi, timestep=0.1, afa=-1.0, gamma1=1.0, gamma2=1.0, Lam=0.004*255**2, mu=1.0, num=0):
    phi = neumann_bound_cond(phi)

    dirac_phi = dirac(phi)
    area_term = afa * dirac_phi * (gamma1 * Energy(img, phi, 0, num=num) - gamma2 * Energy(img, phi, 1, num=num))

    [phi_y, phi_x] = np.gradient(phi)
    s = np.sqrt(np.square(phi_x) + np.square(phi_y))
    n_x = phi_x / (s + 1e-10)
    n_y = phi_y / (s + 1e-10)
    edge_term = Lam * dirac_phi * div(n_x, n_y)

    dist_reg_term = mu * (filters.laplace(phi) - div(n_x, n_y))

    phi += timestep * (area_term + edge_term + dist_reg_term)
    return phi


def read(file, coordinate):
    dataset = gdal.Open(file)
    x_, y_ = coordinate
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    data = dataset.ReadAsArray(x_, y_, width, height)
    del dataset
    return data


Process_img = '../Teacher/Input-A.tiff'
Label = '../Teacher/Label-A.tiff'
F = read(Process_img, (0, 0))
F = np.float64(F)
L = read(Label, (0, 0))
L = np.int64(L)

h, w = F.shape
iter_outer = 300
P = np.ones(shape=[h, w, 8])
for t in range(8):
    for n in range(iter_outer):
        P[:, :, t] = forward(F, P[:, :, t], num=t)

print('SVM')
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(P.reshape(-1, 8), L.flatten())

print('Test')
start = time.time()
pre = clf.predict(P.reshape(-1, 8)).reshape(h, w)
end = time.time()
print(end - start)

k_1 = np.zeros_like(pre)
k_2 = np.zeros_like(pre)
k_3 = np.zeros_like(pre)

k_1[pre == 0] = 1
k_2[pre == 1] = 1
k_3[pre == 2] = 1

print('Noisy:{a}, Clean:{b}, Smooth:{c}'.format(a=k_1.mean().item(), b=k_2.mean().item(), c=k_3.mean().item()))

Mat = Estimate.Confusion_matrix(Pred=pre, Refer=L)
Estimate.Evaluation(Mat, display=True)

plt.imshow(pre, vmin=0, vmax=2, cmap=cmap)
plt.colorbar()
plt.show()
plt.close()

plt.imshow(L, vmin=0, vmax=2, cmap=cmap)
plt.colorbar()
plt.show()
plt.close()

driver = gdal.GetDriverByName('GTiff')
datatype = gdal.GDT_Float64
ori = driver.Create('./Pre.tiff', h, w, 1, datatype)
ori.GetRasterBand(1).WriteArray(np.array(pre))
