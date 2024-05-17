from gausplat import *
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pygausplat as pg
import torchvision
from pytorch_ssim import gau_loss


class GS2DNet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, cov2d, alpha, color):
        global depth
        image, contrib, final_tau, patch_offset_per_tile, gs_id_per_patch =\
            pg.forward(camera.height, camera.width,
                       u, cov2d, alpha, depth, color)
        ctx.save_for_backward(u, cov2d, alpha, color, contrib,
                              final_tau, patch_offset_per_tile, gs_id_per_patch)
        return image

    @staticmethod
    def backward(ctx, dloss_dgammas):
        global depth
        u, cov2d, alpha, color, contrib, \
            final_tau, patch_offset_per_tile, gs_id_per_patch = ctx.saved_tensors
        dloss_dus, dloss_dcov2ds, dloss_dalphas, dloss_dcolors =\
            pg.backward(camera.height, camera.width, u, cov2d, alpha,
                        depth, color, contrib, final_tau,
                        patch_offset_per_tile, gs_id_per_patch, dloss_dgammas)
        return dloss_dus, dloss_dcov2ds, dloss_dalphas, dloss_dcolors


def create_guassian2d_data(camera, gs):
    pw = gs['pos']
    # step1. Transform pw to camera frame,
    # and project it to iamge.
    u, pc = project(pw, camera.Rcw, camera.tcw, camera.K)

    depth = pc[:, 2]

    # step2. Calcuate the 3d Gaussian.
    cov3d = compute_cov_3d(gs['scale'], gs['rot'])

    # step3. Project the 3D Gaussian to 2d image as a 2d Gaussian.
    cov2d = compute_cov_2d(pc, K, cov3d, camera.Rcw)

    # step4. get color info
    ray_dir = pw[:, :3] - camera.cam_center
    ray_dir /= np.linalg.norm(ray_dir, axis=1)[:, np.newaxis]
    color = sh2color(gs['sh'], ray_dir)
    return u, cov2d, gs['alpha'], color, depth


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

    dtypes = [('pos', '<f4', (3,)),
              ('rot', '<f4', (4,)),
              ('scale', '<f4', (3,)),
              ('alpha', '<f4'),
              ('sh', '<f4', (3,))]

    gs = np.frombuffer(gs_data.tobytes(), dtype=dtypes)
    ply_fn = "/home/liu/workspace/gaussian-splatting/output/aecdf69f-c/point_cloud/iteration_10/point_cloud.ply"
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
    focal_x = 1163.2547280302354/2.
    focal_y = 1156.280404988286/2.

    K = np.array([[focal_x, 0, width/2.],
                  [0, focal_y, height/2.],
                  [0, 0, 1.]])

    camera = Camera(id=0, width=width, height=height, K=K, Rcw=Rcw, tcw=tcw)

    u, cov2d, alpha, color, depth = create_guassian2d_data(camera, gs)
    u = torch.from_numpy(u).type(torch.float32).to(device).requires_grad_()
    cov2d = torch.from_numpy(cov2d).type(
        torch.float32).to(device).requires_grad_()
    alpha = torch.from_numpy(alpha).type(
        torch.float32).to(device).requires_grad_()
    depth = torch.from_numpy(depth).type(
        torch.float32).to(device).requires_grad_()
    color = torch.from_numpy(color).type(
        torch.float32).to(device).requires_grad_()

    image, contrib, final_tau, patch_offset_per_tile, gs_id_per_patch =\
        pg.forward(camera.height, camera.width, u, cov2d, alpha, depth, color)

    gs2dnet = GS2DNet

    image_gt = torchvision.io.read_image("imgs/test.png").to(device)
    image_gt = torchvision.transforms.functional.resize(
        image_gt, [height, width]) / 255.

    criterion = nn.MSELoss()
    optimizer = optim.Adam([u, cov2d, alpha, color], lr=0.001, eps=1e-15)
    # image = gs2dnet.apply(u, cov2d, alpha, color)
    # plt.imshow(image.to('cpu').detach().permute(1, 2, 0).numpy())
    # plt.show()
    fig, ax = plt.subplots()
    array = np.zeros(shape=(height, width, 3), dtype=np.uint8)
    im = ax.imshow(array)

    for i in range(1000):
        image = gs2dnet.apply(u, cov2d, alpha, color)
        loss = gau_loss(image, image_gt)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if (i % 10 == 0):
            print("step:%d loss:%f" % (i, loss.item()))
            # plt.imshow(image.to('cpu').detach().permute(1, 2, 0).numpy())
            im_cpu = image.to('cpu').detach().permute(1, 2, 0).numpy()
            im.set_data(im_cpu)
            fig.canvas.flush_events()
            plt.pause(0.1)
    plt.show()
