"""
Pose optimization function with manual quaternion gradients
"""
import torch
import gsplatcu as gsc
from gsplat.utils import *


class PoseFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qcw,
        tcw,
        pws,
        shs,
        alphas,
        scales,
        rots,
        cam,
        optimize_pws=False,
        optimize_rots=False
    ):
        w, x, y, z = qcw

        # Compute rotation matrix from quaternion
        Rcw = torch.stack([
            torch.stack([1.0 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)], dim=0),
            torch.stack([2*(x*y + z*w), 1.0 - 2*(x**2 + z**2), 2*(y*z - x*w)], dim=0),
            torch.stack([2*(x*z - y*w), 2*(y*z + x*w), 1.0 - 2*(x**2 + y**2)], dim=0)
        ], dim=0).to(torch.float32).to('cuda')

        # more detail view forward.pdf
        # step1. Transform pw to camera frame, and project it to image.
        us, pcs, depths, du_dpcs = gsc.project(
            pws, Rcw, tcw, cam.fx, cam.fy, cam.cx, cam.cy, True)

        # step2. Calculate the 3d Gaussian.
        cov3ds, dcov3d_drots, dcov3d_dscales = gsc.computeCov3D(
            rots, scales, depths, True)

        # step3. Calculate the 2d Gaussian.
        cov2ds, dcov2d_dcov3ds, dcov2d_dpcs = gsc.computeCov2D(
            cov3ds, pcs, Rcw, depths, cam.fx, cam.fy, cam.width, cam.height, True)

        # step4. get color info
        twc = -torch.linalg.inv(Rcw) @ tcw
        colors, dcolor_dshs, dcolor_dpws = gsc.sh2Color(shs, pws, twc, True)

        # step5. Blend the 2d Gaussian to image
        cinv2ds, areas, dcinv2d_dcov2ds = gsc.inverseCov2D(cov2ds, depths, True)
        image, contrib, final_tau, patch_range_per_tile, gsid_per_patch =\
            gsc.splat(cam.height, cam.width,
                      us, cinv2ds, alphas, depths, colors, areas)

        # Store the static parameters in the context
        ctx.cam = cam
        ctx.Rcw = Rcw
        ctx.optimize_pws = optimize_pws
        ctx.optimize_rots = optimize_rots

        # Jacobian of rotation matrix w.r.t quaternion [w, x, y, z]
        # Rcw is 3x3, flattened in row-major order
        # dRcw_dq is 9x4 (9 elements of Rcw, 4 elements of quaternion)
        dRcw_dq = torch.stack([
            torch.tensor([0, 0, -4*y, -4*z], dtype=torch.float32, device='cuda'),  # dR00/dq
            torch.tensor([-2*z, 2*y, 2*x, -2*w], dtype=torch.float32, device='cuda'),  # dR01/dq
            torch.tensor([2*y, 2*z, 2*w, 2*x], dtype=torch.float32, device='cuda'),  # dR02/dq
            torch.tensor([2*z, 2*y, 2*x, 2*w], dtype=torch.float32, device='cuda'),  # dR10/dq
            torch.tensor([0, -4*x, 0, -4*z], dtype=torch.float32, device='cuda'),  # dR11/dq
            torch.tensor([-2*x, -2*w, 2*z, 2*y], dtype=torch.float32, device='cuda'),  # dR12/dq
            torch.tensor([-2*y, 2*z, -2*w, 2*x], dtype=torch.float32, device='cuda'),  # dR20/dq
            torch.tensor([2*x, 2*w, 2*z, 2*y], dtype=torch.float32, device='cuda'),  # dR21/dq
            torch.tensor([0, -4*x, -4*y, 0], dtype=torch.float32, device='cuda')  # dR22/dq
        ], dim=0)  # (9, 4)

        pw = pws
        # dpcs_dRcw: how pc changes when Rcw changes
        # pc = Rcw @ pw + tcw
        # dpcs_dRcw[i, :, :] tells how pcs[i] (3D) changes with 9 elements of Rcw
        dpcs_dRcw = torch.zeros((pw.shape[0], 3, 9), dtype=torch.float32, device='cuda')
        dpcs_dRcw[:, 0, 0:3] = pw  # dpc[0]/dRcw[0,:] = pw
        dpcs_dRcw[:, 1, 3:6] = pw  # dpc[1]/dRcw[1,:] = pw
        dpcs_dRcw[:, 2, 6:9] = pw  # dpc[2]/dRcw[2,:] = pw
        
        # Chain rule: dpcs_dqcw = dpcs_dRcw @ dRcw_dq
        dpcs_dqcw = dpcs_dRcw @ dRcw_dq  # (N, 3, 4)

        ctx.save_for_backward(pws, us, cinv2ds, alphas,
                              depths, colors, contrib, final_tau,
                              patch_range_per_tile, gsid_per_patch,
                              dcinv2d_dcov2ds, dcov2d_dcov3ds,
                              dcov3d_drots, dcov3d_dscales, dcolor_dshs,
                              du_dpcs, dcov2d_dpcs, dcolor_dpws, dpcs_dqcw)
        return image

    @staticmethod
    def backward(ctx, dloss_dgammas):
        # Retrieve the saved tensors and static parameters
        cam = ctx.cam
        Rcw = ctx.Rcw
        optimize_pws = ctx.optimize_pws
        optimize_rots = ctx.optimize_rots
        
        pws, us, cinv2ds, alphas, \
            depths, colors, contrib, final_tau, \
            patch_range_per_tile, gsid_per_patch, \
            dcinv2d_dcov2ds, dcov2d_dcov3ds, \
            dcov3d_drots, dcov3d_dscales, dcolor_dshs, \
            du_dpcs, dcov2d_dpcs, dcolor_dpws, dpcs_dqcw = ctx.saved_tensors

        dloss_dus, dloss_dcinv2ds, dloss_dalphas, dloss_dcolors =\
            gsc.splatB(cam.height, cam.width, us, cinv2ds, alphas,
                       depths, colors, contrib, final_tau,
                       patch_range_per_tile, gsid_per_patch, dloss_dgammas)

        dloss_dpcs = dloss_dus @ du_dpcs + dloss_dcinv2ds @ dcinv2d_dcov2ds @ dcov2d_dpcs

        # Camera parameter gradients
        dloss_dqcw = torch.sum(dloss_dpcs @ dpcs_dqcw, dim=0)  # (4,)
        dloss_dtcw = torch.sum(dloss_dpcs, dim=0)  # (3,)

        # Gaussian parameter gradients - only compute if needed
        if optimize_pws:
            dpc_dpws = Rcw
            dloss_dpws = dloss_dus @ du_dpcs @ dpc_dpws + \
                dloss_dcolors @ dcolor_dpws + \
                dloss_dcinv2ds @ dcinv2d_dcov2ds @ dcov2d_dpcs @ dpc_dpws
            dloss_dpws = dloss_dpws.squeeze()
        else:
            dloss_dpws = None

        dloss_dcov2ds = dloss_dcinv2ds @ dcinv2d_dcov2ds
        
        if optimize_rots:
            dloss_drots = dloss_dcov2ds @ dcov2d_dcov3ds @ dcov3d_drots
            dloss_drots = dloss_drots.squeeze()
        else:
            dloss_drots = None
        
        # Always compute these as they are optimized
        dloss_dscales = dloss_dcov2ds @ dcov2d_dcov3ds @ dcov3d_dscales
        dloss_dshs = (dloss_dcolors.permute(0, 2, 1) @
                      dcolor_dshs).permute(0, 2, 1).squeeze()
        dloss_dshs = dloss_dshs.reshape(dloss_dshs.shape[0], -1)

        return dloss_dqcw.squeeze(), \
            dloss_dtcw.squeeze(),\
            dloss_dpws,\
            dloss_dshs,\
            dloss_dalphas.squeeze().unsqueeze(1),\
            dloss_dscales.squeeze(),\
            dloss_drots,\
            None, None, None
