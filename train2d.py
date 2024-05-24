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
from gsnet import GS2DNet
from gsplat.gausplat_dataset import *


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


if __name__ == "__main__":
    path = '/home/liu/bag/gaussian-splatting/tandt/train'
    gs_set = GSplatDataset(path, device='cuda')

    # ply_fn = "/home/liu/workspace/gaussian-splatting/output/train/point_cloud/iteration_10/point_cloud.ply"
    # gs = load_ply(ply_fn)
    gs = gs_set.gs

    cam0, image0_gt = gs_set[0]

    K = np.array([[cam0.fx, 0, cam0.cx],
                  [0, cam0.fy, cam0.cy],
                  [0, 0, 1.]])
    Rcw = cam0.Rcw.cpu().numpy()
    tcw = cam0.tcw.cpu().numpy()
    height, width = cam0.height, cam0.width

    us, cinv2ds, alphas, colors, depths, areas = create_guassian2d_data(
        gs, Rcw, tcw, K)
    us = torch.from_numpy(us).type(torch.float32).to("cuda").requires_grad_()
    cinv2ds = torch.from_numpy(cinv2ds).type(
        torch.float32).to("cuda").requires_grad_()
    alphas = torch.from_numpy(alphas).type(
        torch.float32).to("cuda").requires_grad_()
    depths = torch.from_numpy(depths).type(torch.float32).to("cuda")
    colors = torch.from_numpy(colors).type(
        torch.float32).to("cuda").requires_grad_()
    areas = torch.from_numpy(areas).type(torch.float32).to("cuda")
    image, contrib, final_tau, patch_offset_per_tile, gs_id_per_patch =\
        gsc.splat(height, width, us,
                  cinv2ds, alphas, depths, colors, areas)

    gs2dnet = GS2DNet

    image_gt = torchvision.io.read_image("imgs/test.png").to("cuda")
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
