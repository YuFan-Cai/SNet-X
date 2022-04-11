import os
import torch
import torch.backends.cudnn as cudnn
import model
from osgeo import gdal
import matplotlib.pyplot as plt
import torch.nn.functional as F


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
cudnn.benchmark = True


if __name__ == '__main__':
    M0 = 0
    if M0 == 0:
        model_name = 'model_A.pth'
    else:
        model_name = 'model_B.pth'

    Test_file = '../../Clean.tiff'
    img = gdal.Open(Test_file)
    width = img.RasterXSize
    height = img.RasterYSize
    input = img.ReadAsArray(0, 0, width, height)

    Model_saved = './Weight/'
    Model = model.Apply().to("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(Model_saved + model_name, map_location=torch.device('cpu'))
    model_dict = Model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    Model.load_state_dict(model_dict)

    Model.eval()
    with torch.no_grad():
        input = torch.tensor(input).to("cuda" if torch.cuda.is_available() else "cpu").float().unsqueeze(dim=0)

        pre = Model(input)
        pre = torch.max(F.softmax(pre, dim=1), dim=1)[1]

        k_1 = torch.zeros_like(pre, dtype=torch.float)
        k_2 = torch.zeros_like(pre, dtype=torch.float)
        k_3 = torch.zeros_like(pre, dtype=torch.float)

        k_1[pre == 0] = 1
        k_2[pre == 1] = 1
        k_3[pre == 2] = 1

        print('Noisy:{a}, Clean:{b}, Smooth:{c}'.format(a=k_1.mean().item(), b=k_2.mean().item(), c=k_3.mean().item()))

        plt.imshow(pre[0, :, :].cpu(), vmin=0, vmax=2)
        plt.colorbar()
        plt.show()
        plt.close()