import torch
import torchvision
import numpy as np
from math import exp
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

device='cuda'


# https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
SH_C0_0 = 0.28209479177387814  # Y0,0:  1/2*sqrt(1/pi)       plus

SH_C1_0 = -0.4886025119029199  # Y1,-1: sqrt(3/(4*pi))       minus
SH_C1_1 = 0.4886025119029199   # Y1,0:  sqrt(3/(4*pi))       plus
SH_C1_2 = -0.4886025119029199  # Y1,1:  sqrt(3/(4*pi))       minus

SH_C2_0 = 1.0925484305920792   # Y2,-2: 1/2 * sqrt(15/pi)    plus
SH_C2_1 = -1.0925484305920792  # Y2,-1: 1/2 * sqrt(15/pi)    minus
SH_C2_2 = 0.31539156525252005  # Y2,0:  1/4*sqrt(5/pi)       plus
SH_C2_3 = -1.0925484305920792  # Y2,1:  1/2*sqrt(15/pi)      minus
SH_C2_4 = 0.5462742152960396   # Y2,2:  1/4*sqrt(15/pi)      plus

SH_C3_0 = -0.5900435899266435  # Y3,-3: 1/4*sqrt(35/(2*pi))  minus
SH_C3_1 = 2.890611442640554    # Y3,-2: 1/2*sqrt(105/pi)     plus
SH_C3_2 = -0.4570457994644658  # Y3,-1: 1/4*sqrt(21/(2*pi))  minus
SH_C3_3 = 0.3731763325901154   # Y3,0:  1/4*sqrt(7/pi)       plus
SH_C3_4 = -0.4570457994644658  # Y3,1:  1/4*sqrt(21/(2*pi))  minus
SH_C3_5 = 1.445305721320277    # Y3,2:  1/4*sqrt(105/pi)     plus
SH_C3_6 = -0.5900435899266435  # Y3,3:  1/4*sqrt(35/(2*pi))  minus

SH_C4_0 = 2.5033429417967046  # Y4,-4:  3/4*sqrt(35/pi)       plus
SH_C4_1 = -1.7701307697799304  # Y4,-3:  3/4*sqrt(35/(2*pi))  minus
SH_C4_2 = 0.9461746957575601  # Y4,-2:  3/4*sqrt(5/pi)        plus
SH_C4_3 = -0.6690465435572892  # Y4,-1:  3/4*sqrt(5/(2*pi))   minus
SH_C4_4 = 0.10578554691520431  # Y4,0:  3/16*sqrt(1/pi)       plus
SH_C4_5 = -0.6690465435572892  # Y4,1:  3/4*sqrt(5/(2*pi))    minus
SH_C4_6 = 0.47308734787878004  # Y4,2:  3/8*sqrt(5/pi)        plus
SH_C4_7 = -1.7701307697799304  # Y4,3:  3/4*sqrt(35/(2*pi))   minus
SH_C4_8 = 0.6258357354491761  # Y4,4:  3/16*sqrt(35/pi)       plus


def sh2color(sh, ray_dir):
    dCdSH = torch.zeros([sh.shape[0], ray_dir.shape[0]], dtype=torch.float32, device=device)
    sh_dim = sh.shape[0]
    dCdSH[0] = SH_C0_0
    color = (dCdSH[0] * sh[0][:, np.newaxis]) + 0.5

    if (sh_dim <= 1):
        return color, dCdSH.T

    x = ray_dir[:, 0]
    y = ray_dir[:, 1]
    z = ray_dir[:, 2]
    dCdSH[1] = SH_C1_0 * y
    dCdSH[2] = SH_C1_1 * z
    dCdSH[3] = SH_C1_2 * x
    color = color + \
        dCdSH[1] * sh[1][:, np.newaxis] + \
        dCdSH[2] * sh[2][:, np.newaxis] + \
        dCdSH[3] * sh[3][:, np.newaxis]

    if (sh_dim <= 4):
        return color, dCdSH.T
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    yz = y * z
    xz = x * z
    dCdSH[4] = SH_C2_0 * xy
    dCdSH[5] = SH_C2_1 * yz
    dCdSH[6] = SH_C2_2 * (2.0 * zz - xx - yy)
    dCdSH[7] = SH_C2_3 * xz
    dCdSH[8] = SH_C2_4 * (xx - yy)
    color = color + \
        dCdSH[4] * sh[4][:, np.newaxis] + \
        dCdSH[5] * sh[5][:, np.newaxis] + \
        dCdSH[6] * sh[6][:, np.newaxis] + \
        dCdSH[7] * sh[7][:, np.newaxis] + \
        dCdSH[8] * sh[8][:, np.newaxis]

    if (sh_dim <= 9):
        return color, dCdSH.T
    dCdSH[9 ] = SH_C3_0 * y * (3.0 * xx - yy)
    dCdSH[10] = SH_C3_1 * xy * z
    dCdSH[11] = SH_C3_2 * y * (4.0 * zz - xx - yy)
    dCdSH[12] = SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)
    dCdSH[13] = SH_C3_4 * x * (4.0 * zz - xx - yy)
    dCdSH[14] = SH_C3_5 * z * (xx - yy)
    dCdSH[15] = SH_C3_6 * x * (xx - 3.0 * yy)

    color = color +  \
        dCdSH[9 ] * sh[9 ][:, np.newaxis] + \
        dCdSH[10] * sh[10][:, np.newaxis] + \
        dCdSH[11] * sh[11][:, np.newaxis] + \
        dCdSH[12] * sh[12][:, np.newaxis] + \
        dCdSH[13] * sh[13][:, np.newaxis] + \
        dCdSH[14] * sh[14][:, np.newaxis] + \
        dCdSH[15] * sh[15][:, np.newaxis]

    if (sh_dim <= 16):
        return color, dCdSH.T
    dCdSH[16] = SH_C4_0 * xy * (xx - yy)
    dCdSH[17] = SH_C4_1 * y * (3*xx - yy)*z
    dCdSH[18] = SH_C4_2 * xy * (7*zz - 1)
    dCdSH[19] = SH_C4_3 * y * (7*zz*z - 3*z)  # 4*zz*z - 3*xx*z - 3*z*yy
    dCdSH[20] = SH_C4_4 * x * (35 * zz * zz - 30 * zz + 3)
    dCdSH[21] = SH_C4_5 * z * x * (7 * zz * z - 3*z)
    dCdSH[22] = SH_C4_6 * (xx - yy) * (7*zz-1)
    dCdSH[23] = SH_C4_7 * x * (xx - 3*yy)*z
    dCdSH[24] = SH_C4_8 * xx * (xx - 3* yy) - yy*(3*xx - yy) 

    color = color +  \
        dCdSH[16] * sh[9 ][:, np.newaxis] + \
        dCdSH[17] * sh[10][:, np.newaxis] + \
        dCdSH[18] * sh[11][:, np.newaxis] + \
        dCdSH[19] * sh[12][:, np.newaxis] + \
        dCdSH[20] * sh[13][:, np.newaxis] + \
        dCdSH[21] * sh[14][:, np.newaxis] + \
        dCdSH[22] * sh[15][:, np.newaxis]
    return color, dCdSH.T



# spherical harmonics
class SHNet(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sh):
        color, dCdSH = sh2color(sh, xyz)
        ctx.save_for_backward(dCdSH)
        return color.reshape(3, H, W)

    @staticmethod
    def backward(ctx, dL_dC):  # dL_dy = dL/dy
        dCdSH,  = ctx.saved_tensors
        dLdSH = dL_dC.reshape(3, -1) @ dCdSH
        return dLdSH.T

def d_tanh(x):
    return 1 / (x.cosh() ** 2)

if __name__ == "__main__":
    sh = np.zeros([16, 3], dtype=np.float32) # 1. 4, 9, 16, 25
    sh = torch.from_numpy(sh).to(device).requires_grad_()

    W = int(979)  # 1957  # 979
    H = int(546)  # 1091  # 546

    theta = torch.linspace(0, torch.pi, H, dtype=torch.float32, device=device)
    phi = torch.linspace(0, 2 * torch.pi, W, dtype=torch.float32, device=device)
    angle = torch.stack((torch.meshgrid(theta, phi)), axis=2)
    x = torch.sin(angle[:,:,0]) * torch.cos(angle[:,:,1])
    y = torch.sin(angle[:,:,0]) * torch.sin(angle[:,:,1])
    z = torch.cos(angle[:,:,0])
    xyz = torch.dstack((x, y, z)).reshape(-1, 3)

    shnet = SHNet

    image_gt = torchvision.io.read_image("imgs/Solarsystemscope_texture_8k_earth_daymap.jpg").to(device)
    image_gt = torchvision.transforms.functional.resize(image_gt, [H, W]) / 255.

    criterion = nn.MSELoss()
    optimizer = optim.SGD([sh], lr=1.)


    lambda_dssim = 0.2
    learning_rate = 1.
    for i in range(100):
        image = shnet.apply(sh)
        loss = criterion(image, image_gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
    image = shnet.apply(sh)

    # Display the ground truth image
    plt.subplot(1, 2, 1)
    plt.imshow(image_gt.to('cpu').detach().permute(1, 2, 0).numpy())
    plt.title('ground truth image')

    # Display the sh image
    plt.subplot(1, 2, 2)
    plt.imshow(image.to('cpu').detach().permute(1, 2, 0).numpy())
    plt.title('sh image')

    plt.show()

