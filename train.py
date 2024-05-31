import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from gsplat.pytorch_ssim import gau_loss
from gsplat.gau_io import *
from gsplat.gausplat_dataset import *
from gsnet import GSNet, logit
from random import randint
import time
import gsplatcu as gsc


class GSModel(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pws,
        shs,
        alphas,
        scales,
        rots,
        cam,
    ):
        # step1. Transform pw to camera frame,
        # and project it to iamge.
        us, pcs, depths, du_dpcs = gsc.project(
            pws, cam.Rcw, cam.tcw, cam.fx, cam.fy, cam.cx, cam.cy, True)

        # step2. Calcuate the 3d Gaussian.
        cov3ds, dcov3d_drots, dcov3d_dscales = gsc.computeCov3D(
            rots, scales, depths, True)

        # step3. Calcuate the 2d Gaussian.
        cov2ds, dcov2d_dcov3ds, dcov2d_dpcs = gsc.computeCov2D(
            cov3ds, pcs, cam.Rcw, depths, cam.fx, cam.fy, cam.width, cam.height, True)

        # step4. get color info
        colors, dcolor_dshs, dcolor_dpws = gsc.sh2Color(shs, pws, cam.twc, True)

        # step5. Blend the 2d Gaussian to image
        cinv2ds, areas, dcinv2d_dcov2ds = gsc.inverseCov2D(cov2ds, depths, True)
        image, contrib, final_tau, patch_range_per_tile, gsid_per_patch =\
            gsc.splat(cam.height, cam.width,
                      us, cinv2ds, alphas, depths, colors, areas)

        # Keep relevant tensors for backward
        ctx.cam = cam
        ctx.save_for_backward(us, cinv2ds, alphas,
                              depths, colors, contrib, final_tau,
                              patch_range_per_tile, gsid_per_patch,
                              dcinv2d_dcov2ds, dcov2d_dcov3ds,
                              dcov3d_drots, dcov3d_dscales, dcolor_dshs,
                              du_dpcs, dcov2d_dpcs, dcolor_dpws)
        return image

    @staticmethod
    def backward(ctx, dloss_dgammas):
        # Restore necessary values from context
        cam = ctx.cam
        us, cinv2ds, alphas, \
            depths, colors, contrib, final_tau,\
            patch_range_per_tile, gsid_per_patch,\
            dcinv2d_dcov2ds, dcov2d_dcov3ds,\
            dcov3d_drots, dcov3d_dscales, dcolor_dshs,\
            du_dpcs, dcov2d_dpcs, dcolor_dpws = ctx.saved_tensors

        dloss_dus, dloss_dcinv2ds, dloss_dalphas, dloss_dcolors =\
            gsc.splatB(cam.height, cam.width, us, cinv2ds, alphas,
                       depths, colors, contrib, final_tau,
                       patch_range_per_tile, gsid_per_patch, dloss_dgammas)

        dpc_dpws = cam.Rcw

        dloss_drots = dloss_dcinv2ds @ dcinv2d_dcov2ds @ dcov2d_dcov3ds @ dcov3d_drots
        dloss_dscales = dloss_dcinv2ds @ dcinv2d_dcov2ds @ dcov2d_dcov3ds @ dcov3d_dscales

        dloss_dshs = (dloss_dcolors.permute(0, 2, 1) @
                      dcolor_dshs).permute(0, 2, 1).squeeze()
        dloss_dpws = dloss_dus @ du_dpcs @ dpc_dpws + \
            dloss_dcolors @ dcolor_dpws + \
            dloss_dcinv2ds @ dcinv2d_dcov2ds @ dcov2d_dpcs @ dpc_dpws

        return dloss_dpws.squeeze(),\
            dloss_dshs.squeeze(),\
            dloss_dalphas.squeeze().unsqueeze(1),\
            dloss_dscales.squeeze(),\
            dloss_drots.squeeze(),\
            None


if __name__ == "__main__":
    path = '/home/liu/bag/gaussian-splatting/tandt/train'
    gs_set = GSplatDataset(path)

    gsnet = GSNet

    gs = gs_set.gs

    pws = torch.from_numpy(gs['pw']).type(
        torch.float32).to('cuda').requires_grad_()
    rots_raw = torch.from_numpy(gs['rot']).type(
        # the unactivated scales
        torch.float32).to('cuda').requires_grad_()
    scales_raw = torch.log(torch.from_numpy(gs['scale']).type(
        torch.float32).to('cuda')).requires_grad_()
    # the unactivated alphas
    alphas_raw = logit(torch.from_numpy(gs['alpha'][:, np.newaxis]).type(
        torch.float32).to('cuda')).requires_grad_()

    shs = torch.from_numpy(gs['sh']).type(
        torch.float32).to('cuda').requires_grad_()

    l = [
        {'params': [rots_raw], 'lr': 0.001, "name": "rot"},
        {'params': [scales_raw], 'lr': 0.005, "name": "scale"},
        {'params': [shs], 'lr': 0.001, "name": "sh"},
        {'params': [alphas_raw], 'lr': 0.05, "name": "alpha"},
        {'params': [pws], 'lr': 0.001, "name": "pw"}
    ]

    optimizer = optim.Adam(l, lr=0.000, eps=1e-15)

    cam0, _ = gs_set[0]
    fig, ax = plt.subplots()
    array = np.zeros(shape=(cam0.height, cam0.width, 3), dtype=np.uint8)
    im = ax.imshow(array)

    n_epochs = 100
    n = len(gs_set)
    # n = 1
    for epoch in range(n_epochs):
        idxs = np.arange(n)
        np.random.shuffle(idxs)
        avg_loss = 0
        for i in idxs:
            cam, image_gt = gs_set[i]
            # Limit the value of alphas: 0 < alphas < 1
            alphas = torch.sigmoid(alphas_raw)
            # Limit the value of scales > 0
            scales = torch.exp(scales_raw)
            # Limit the value of rot, normal of rots is 1
            rots = torch.nn.functional.normalize(rots_raw)

            image = GSModel.apply(
                pws,
                shs,
                alphas,
                scales,
                rots,
                cam,
            )

            loss = gau_loss(image, image_gt)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            avg_loss += loss.item()
            if (i == 0):
                im_cpu = image.to('cpu').detach().permute(1, 2, 0).numpy()
                im.set_data(im_cpu)
                fig.canvas.flush_events()
                plt.pause(0.1)
                # plt.show()
        avg_loss = avg_loss / n
        print("epoch:%d avg_loss:%f" % (epoch, avg_loss))
        # save_torch_params("tmp.npy", rots, scales, shs.reshape([shs.shape[0], -1]), alphas.reshape(-1), pws)
        # if (epoch % 10 == 0):
        #    save_torch_params("epoch%04d.npy" % epoch, rots, scales, shs, alphas, pws)
