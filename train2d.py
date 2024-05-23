import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import gsplatcu as gsc
import torchvision
from gsplat.pytorch_ssim import gau_loss
from gsplat.read_ply import *
from gsplat.gausplat import *


class GS2DNet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, us, cinv2ds, alphas, colors, depths, areas, height, width):
        # Store the static parameters in the context
        ctx.depths = depths
        ctx.height = height
        ctx.width = width
        image, contrib, final_tau, patch_offset_per_tile, gs_id_per_patch =\
            gsc.splat(height, width,
                      us, cinv2ds, alphas, depths, colors, areas)
        ctx.save_for_backward(us, cinv2ds, alphas, colors, contrib,
                              final_tau, patch_offset_per_tile, gs_id_per_patch)
        return image

    @staticmethod
    def backward(ctx, dloss_dgammas):
        us, cinv2ds, alphas, colors, contrib, \
            final_tau, patch_offset_per_tile, gs_id_per_patch = ctx.saved_tensors
        # Retrieve the saved tensors and static parameters
        depths = ctx.depths
        height = ctx.height
        width = ctx.width
        dloss_dus, dloss_dcinv2ds, dloss_dalphas, dloss_dcolors =\
            gsc.splatB(height, width, us, cinv2ds, alphas,
                       depths, colors, contrib, final_tau,
                       patch_offset_per_tile, gs_id_per_patch, dloss_dgammas)
        return dloss_dus.squeeze(), dloss_dcinv2ds.squeeze(), dloss_dalphas.squeeze(),\
            dloss_dcolors.squeeze(), None, None, None, None


def create_guassian2d_data(gs, Rcw, tcw, K):
    pws = gs['pw']
    # step1. Transform pw to camera frame,
    # and project it to iamge.
    us, pcs = project(pws, Rcw, tcw, K)

    depths = pcs[:, 2]

    # step2. Calcuate the 3d Gaussian.
    cov3ds = compute_cov_3d(gs['scale'], gs['rot'])

    # step3. Project the 3D Gaussian to 2d image as a 2d Gaussian.
    cov2ds = compute_cov_2d(pcs, K, cov3ds, Rcw)

    cinv2ds, areas = inverse_cov2d(cov2ds)

    # step4. get color info
    colors = sh2color(gs['sh'], pws, twc=np.linalg.inv(Rcw) @ (-tcw))
    return us, cinv2ds, gs['alpha'], colors, depths, areas


device = 'cuda'


if __name__ == "__main__":
    gs_data = np.array([[0.,  0.,  0.,  # xyz
                        1.,  0.,  0., 0.,  # rot
                        0.5,  0.5,  0.5,  # size
                        1.,
                        1.772484,  -1.772484,  1.772484],
                        [1.,  0.,  0.,
                        1.,  0.,  0., 0.,
                        2,  0.5,  0.5,
                        1.,
                        1.772484,  -1.772484, -1.772484],
                        [0.,  1.,  0.,
                        1.,  0.,  0., 0.,
                        0.5,  2,  0.5,
                        1.,
                        -1.772484, 1.772484, -1.772484],
                        [0.,  0.,  1.,
                        1.,  0.,  0., 0.,
                        0.5,  0.5,  2,
                        1.,
                        -1.772484, -1.772484,  1.772484]
                        ], dtype=np.float32)

    dtypes = [('pw', '<f4', (3,)),
              ('rot', '<f4', (4,)),
              ('scale', '<f4', (3,)),
              ('alpha', '<f4'),
              ('sh', '<f4', (3,))]

    gs = np.frombuffer(gs_data.tobytes(), dtype=dtypes)
    ply_fn = "/home/liu/workspace/gaussian-splatting/output/train/point_cloud/iteration_10/point_cloud.ply"
    gs = load_ply(ply_fn)

    # Camera info
    tcw = np.array([1.03796196, 0.42017467, 4.67804612])
    Rcw = np.array([[0.89699204,  0.06525223,  0.43720409],
                    [-0.04508268,  0.99739184, -0.05636552],
                    [-0.43974177,  0.03084909,  0.89759429]]).T

    # width = int(32)
    # height = int(16)
    # focal_x = 16
    # focal_y = 16

    width = int(979)
    height = int(546)
    focal_x = 581.6273640151177
    focal_y = 578.140202494143

    K = np.array([[focal_x, 0, width/2.],
                  [0, focal_y, height/2.],
                  [0, 0, 1.]])

    us, cinv2ds, alphas, colors, depths, areas = create_guassian2d_data(
        gs, Rcw, tcw, K)
    us = torch.from_numpy(us).type(torch.float32).to(device).requires_grad_()
    cinv2ds = torch.from_numpy(cinv2ds).type(
        torch.float32).to(device).requires_grad_()
    alphas = torch.from_numpy(alphas).type(
        torch.float32).to(device).requires_grad_()
    depths = torch.from_numpy(depths).type(torch.float32).to(device)
    colors = torch.from_numpy(colors).type(
        torch.float32).to(device).requires_grad_()
    areas = torch.from_numpy(areas).type(torch.float32).to(device)
    image, contrib, final_tau, patch_offset_per_tile, gs_id_per_patch =\
        gsc.splat(height, width, us,
                  cinv2ds, alphas, depths, colors, areas)

    gs2dnet = GS2DNet

    image_gt = torchvision.io.read_image("imgs/test.png").to(device)
    image_gt = torchvision.transforms.functional.resize(
        image_gt, [height, width]) / 255.

    optimizer = optim.Adam([us, cinv2ds, alphas, colors], lr=0.005, eps=1e-15)

    fig, ax = plt.subplots()
    array = np.zeros(shape=(height, width, 3), dtype=np.uint8)
    im = ax.imshow(array)

    for i in range(100):
        image = gs2dnet.apply(us, cinv2ds, alphas, colors,
                              depths, areas, height, width)
        loss = gau_loss(image, image_gt)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if (i % 1 == 0):
            print("step:%d loss:%f" % (i, loss.item()))
            im_cpu = image.to('cpu').detach().permute(1, 2, 0).numpy()
            im.set_data(im_cpu)
            plt.pause(0.1)
    plt.show()
