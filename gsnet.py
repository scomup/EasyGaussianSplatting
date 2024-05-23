import torch
import gsplatcu as gsc


class GSNet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rots, scales, shs, alphas, pws, camera):
        # Store the static parameters in the context
        ctx.camera = camera

        global Rcw, tcw, twc, fx, fy, cx, cy, height, width
        # step1. Transform pw to camera frame,
        # and project it to iamge.
        us, pcs, du_dpcs = gsc.project(
            pws, camera.Rcw, camera.tcw, camera.fx, camera.fy, camera.cx, camera.cy, True)
        depths = pcs[:, 2]

        # step2. Calcuate the 3d Gaussian.
        cov3ds, dcov3d_drots, dcov3d_dscales = gsc.computeCov3D(
            rots, scales, True)

        # step3. Calcuate the 2d Gaussian.
        cov2ds, dcov2d_dcov3ds, dcov2d_dpcs = gsc.computeCov2D(
            cov3ds, pcs, camera.Rcw, camera.fx, camera.fy, True)

        # step4. get color info
        colors, dcolor_dshs, dcolor_dpws = gsc.sh2Color(shs, pws, camera.twc, True)

        # step5. Blend the 2d Gaussian to image
        cinv2ds, areas, dcinv2d_dcov2ds = gsc.inverseCov2D(cov2ds, True)
        image, contrib, final_tau, patch_offset_per_tile, gs_id_per_patch =\
            gsc.splat(camera.height, camera.width,
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
        # Retrieve the saved tensors and static parameters
        camera = ctx.camera

        contrib, final_tau, \
            patch_offset_per_tile, gs_id_per_patch, \
            cinv2ds, dcinv2d_dcov2ds, \
            colors, dcolor_dshs, dcolor_dpws, \
            dcov2d_dcov3ds, dcov2d_dpcs, \
            dcov3d_drots, dcov3d_dscales, \
            depths, us, du_dpcs, alphas = ctx.saved_tensors

        # backward process of splat
        dloss_dus, dloss_dcinv2ds, dloss_dalphas, dloss_dcolors =\
            gsc.splatB(camera.height, camera.width, us, cinv2ds, alphas,
                       depths, colors, contrib, final_tau,
                       patch_offset_per_tile, gs_id_per_patch, dloss_dgammas)

        dpc_dpws = camera.Rcw
        dloss_dcov2ds = dloss_dcinv2ds @ dcinv2d_dcov2ds

        # calcuate all jacobians
        dloss_drots = dloss_dcov2ds @ dcov2d_dcov3ds @ dcov3d_drots
        dloss_dscales = dloss_dcov2ds @ dcov2d_dcov3ds @ dcov3d_dscales
        dloss_dshs = (dloss_dcolors.permute(0, 2, 1) @ dcolor_dshs)\
            .permute(0, 2, 1).reshape(dloss_dcolors.shape[0], 1, -1)
        dloss_dalphas = dloss_dalphas
        dloss_dpws = dloss_dus @ du_dpcs @ dpc_dpws + \
            dloss_dcolors @ dcolor_dpws + \
            dloss_dcov2ds @ dcov2d_dpcs @ dpc_dpws
        return dloss_drots.squeeze(), dloss_dscales.squeeze(), \
            dloss_dshs.squeeze(), dloss_dalphas.squeeze(), \
            dloss_dpws.squeeze(), None
