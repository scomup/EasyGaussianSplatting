from gausplat import *
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import pygausplat as pg
import torchvision
from pytorch_ssim import gau_loss


class GSNet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rots, scales, shs, alphas, pws):
        
        global Rcw, tcw, twc, focal_x, focal_y, center_x, center_y, height, width

        us, pcs, du_dpcs = pg.project(pws, Rcw, tcw, focal_x, focal_y, center_x, center_y, True)
        depths = pcs[:, 2]

        # step2. Calcuate the 3d Gaussian.
        cov3ds, dcov3d_drots, dcov3d_dscales = pg.computeCov3D(rots, scales, True)

        # step3. Calcuate the 2d Gaussian.
        cov2ds, dcov2d_dcov3ds, dcov2d_dpcs = pg.computeCov2D(cov3ds, pcs, Rcw, focal_x, focal_y, True)

        # step4. get color info
        colors, dcolor_dshs, dcolor_dpws = pg.sh2Color(shs, pws, twc, True)

        # step5. Blend the 2d Gaussian to image
        cinv2ds, areas, dcinv2d_dcov2ds = pg.inverseCov2D(cov2ds, True)
        image, contrib, final_tau, patch_offset_per_tile, gs_id_per_patch =\
            pg.splat(height, width,
                     us, cinv2ds, alphas, depths, colors, areas)
    
        ctx.save_for_backward(contrib, final_tau, 
                              patch_offset_per_tile, gs_id_per_patch,
                              cinv2ds, dcinv2d_dcov2ds,
                              colors, dcolor_dshs, dcolor_dpws,
                              dcov2d_dcov3ds, dcov2d_dpcs,
                              dcov3d_drots, dcov3d_dscales,
                              depths, us, du_dpcs, alphas)
        return image

    @staticmethod
    def backward(ctx, dloss_dgammas):
        global Rcw, tcw, height, width
        contrib, final_tau,\
            patch_offset_per_tile, gs_id_per_patch,\
            cinv2ds, dcinv2d_dcov2ds,\
            colors, dcolor_dshs, dcolor_dpws,\
            dcov2d_dcov3ds, dcov2d_dpcs,\
            dcov3d_drots, dcov3d_dscales,\
            depths, us, du_dpcs, alphas = ctx.saved_tensors
    
        dloss_dus, dloss_dcinv2ds, dloss_dalphas, dloss_dcolors =\
        pg.splatB(height, width, us, cinv2ds, alphas,\
            depths, colors, contrib, final_tau,\
            patch_offset_per_tile, gs_id_per_patch, dloss_dgammas)
        
        dpc_dpws = Rcw
        dloss_dcov2ds = dloss_dcinv2ds @ dcinv2d_dcov2ds
    
        dloss_drots = dloss_dcov2ds @ dcov2d_dcov3ds @ dcov3d_drots
        dloss_dscales = dloss_dcov2ds @ dcov2d_dcov3ds @ dcov3d_dscales
        dloss_dshs = (dloss_dcolors.permute(0, 2, 1) @ dcolor_dshs)\
            .permute(0, 2, 1).reshape(dloss_dcolors.shape[0], 1, -1)
        dloss_dalphas = dloss_dalphas
        dloss_dpws = dloss_dus @ du_dpcs @ dpc_dpws + \
            dloss_dcolors @ dcolor_dpws + \
            dloss_dcov2ds @ dcov2d_dpcs @ dpc_dpws    
        return dloss_drots.squeeze(), dloss_dscales.squeeze(),\
            dloss_dshs.squeeze(), dloss_dalphas.squeeze(), dloss_dpws.squeeze()


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
    ply_fn = "/home/liu/workspace/gaussian-splatting/output/train2d/point_cloud/iteration_10/point_cloud.ply"
    gs = load_ply(ply_fn)

    # Camera info
    tcw = np.array([1.03796196, 0.42017467, 4.67804612])
    Rcw = np.array([[0.89699204,  0.06525223,  0.43720409],
                    [-0.04508268,  0.99739184, -0.05636552],
                    [-0.43974177,  0.03084909,  0.89759429]]).T

    width = int(979)
    height = int(546)
    focal_x = 581.6273640151177
    focal_y = 578.140202494143
    center_x = width/2.
    center_y = height/2.

    twc = np.linalg.inv(Rcw) @ (-tcw)
    pws = torch.from_numpy(gs['pos']).type(torch.float32).to('cuda').requires_grad_()
    rots = torch.from_numpy(gs['rot']).type(torch.float32).to('cuda').requires_grad_()
    scales = torch.from_numpy(gs['scale']).type(torch.float32).to('cuda').requires_grad_()
    alphas = torch.from_numpy(gs['alpha']).type(torch.float32).to('cuda').requires_grad_()
    shs = torch.from_numpy(gs['sh']).type(torch.float32).to('cuda').requires_grad_()
    Rcw = torch.from_numpy(Rcw).type(torch.float32).to('cuda')
    tcw = torch.from_numpy(tcw).type(torch.float32).to('cuda')
    twc = torch.from_numpy(twc).type(torch.float32).to('cuda')

    gsnet = GSNet

    image_gt = torchvision.io.read_image("imgs/test.png").to(device)
    image_gt = torchvision.transforms.functional.resize(
        image_gt, [height, width]) / 255.

    optimizer = optim.Adam([rots, scales, shs, alphas, pws], lr=0.001, eps=1e-15)

    fig, ax = plt.subplots()
    array = np.zeros(shape=(height, width, 3), dtype=np.uint8)
    im = ax.imshow(array)

    for i in range(100):
        image = gsnet.apply(rots, scales, shs, alphas, pws)
        loss = gau_loss(image, image_gt)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if (i % 1 == 0):
            print("step:%d loss:%f" % (i, loss.item()))
            im_cpu = image.to('cpu').detach().permute(1, 2, 0).numpy()
            im.set_data(im_cpu)
            fig.canvas.flush_events()
            plt.pause(0.1)
    plt.show()
