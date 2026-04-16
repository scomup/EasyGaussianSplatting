"""
Simplified Pose Function for isotropic Gaussians
- No rotation (identity quaternion)
- Fixed position (pws)
- Only optimizes: camera pose, colors, alphas, scales
"""
import torch
import gsplatcu as gsc


class SimplePoseFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qcw,
        tcw,
        pws,       # FIXED - no gradient
        colors,    # RGB colors
        alphas,
        scales,    # Isotropic scale (N,) or (N,1)
        cam
    ):
        w, x, y, z = qcw

        # Compute rotation matrix from quaternion
        Rcw = torch.stack([
            torch.stack([1.0 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)], dim=0),
            torch.stack([2*(x*y + z*w), 1.0 - 2*(x**2 + z**2), 2*(y*z - x*w)], dim=0),
            torch.stack([2*(x*z - y*w), 2*(y*z + x*w), 1.0 - 2*(x**2 + y**2)], dim=0)
        ], dim=0).to(torch.float32).to('cuda')

        # Convert isotropic scales to 3D
        if scales.dim() == 1:
            scales = scales.unsqueeze(1)
        scales_3d = scales.repeat(1, 3)  # (N, 3) same scale in all directions
        
        # Identity quaternions for rotation (no rotation)
        N = pws.shape[0]
        rots = torch.zeros((N, 4), dtype=torch.float32, device='cuda')
        rots[:, 0] = 1.0  # [w=1, x=0, y=0, z=0]

        # Convert RGB to SH
        C0 = 0.28209479177387814
        sh_dc = (colors - 0.5) / C0
        
        # Combine with zero high-order SH
        shs = sh_dc  # Only use DC component for speed
        
        # Step1: Transform pw to camera frame, and project it to image
        us, pcs, depths, du_dpcs = gsc.project(
            pws, Rcw, tcw, cam.fx, cam.fy, cam.cx, cam.cy, True)

        # Step2: Calculate the 3d Gaussian (isotropic, no rotation effect)
        # Note: For isotropic Gaussians, computeCov2D handles the covariance directly
        # We don't actually need computeCov3D for the forward pass
        # cov3ds, dcov3d_dscales = gsc.computeCov3D(
        #     rots, scales_3d, depths, True)

        # Step3: Calculate the 2d Gaussian
        # Pass scales_3d directly (first column contains the isotropic scale)
        scales_iso = scales_3d[:, 0]  # Extract isotropic scale (all 3 columns are same)
        cov2ds, dcov2d_dscales, dcov2d_dpcs = gsc.computeCov2D(
            scales_iso, pcs, Rcw, depths, cam.fx, cam.fy, cam.width, cam.height, True)

        # Step4: Get color info
        twc = -torch.linalg.inv(Rcw) @ tcw
        # For simple color, we don't need view-dependent effects
        # Just use the DC component directly
        color_output = torch.clamp(colors, 0, 1)
        
        # Step5: Blend the 2d Gaussian to image
        cinv2ds, areas, dcinv2d_dcov2ds = gsc.inverseCov2D(cov2ds, depths, True)
        image, contrib, final_tau, patch_range_per_tile, gsid_per_patch =\
            gsc.splat(cam.height, cam.width,
                      us, cinv2ds, alphas, depths, color_output, areas)

        # Store for backward
        ctx.cam = cam
        ctx.Rcw = Rcw

        # Jacobian of rotation matrix w.r.t quaternion
        dRcw_dq = torch.stack([
            torch.tensor([0, 0, -4*y, -4*z], dtype=torch.float32, device='cuda'),
            torch.tensor([-2*z, 2*y, 2*x, -2*w], dtype=torch.float32, device='cuda'),
            torch.tensor([2*y, 2*z, 2*w, 2*x], dtype=torch.float32, device='cuda'),
            torch.tensor([2*z, 2*y, 2*x, 2*w], dtype=torch.float32, device='cuda'),
            torch.tensor([0, -4*x, 0, -4*z], dtype=torch.float32, device='cuda'),
            torch.tensor([-2*x, -2*w, 2*z, 2*y], dtype=torch.float32, device='cuda'),
            torch.tensor([-2*y, 2*z, -2*w, 2*x], dtype=torch.float32, device='cuda'),
            torch.tensor([2*x, 2*w, 2*z, 2*y], dtype=torch.float32, device='cuda'),
            torch.tensor([0, -4*x, -4*y, 0], dtype=torch.float32, device='cuda')
        ], dim=0)

        dpcs_dRcw = torch.zeros((pws.shape[0], 3, 9), dtype=torch.float32, device='cuda')
        dpcs_dRcw[:, 0, 0:3] = pws
        dpcs_dRcw[:, 1, 3:6] = pws
        dpcs_dRcw[:, 2, 6:9] = pws
        dpcs_dqcw = dpcs_dRcw @ dRcw_dq

        # dcov2d_dscales is (N, 3, 1) so we can reshape to (N, 3)
        dcov2d_dscales_reshaped = dcov2d_dscales.squeeze(-1)

        ctx.save_for_backward(us, cinv2ds, alphas,
                              depths, color_output, contrib, final_tau,
                              patch_range_per_tile, gsid_per_patch,
                              dcinv2d_dcov2ds, dcov2d_dscales_reshaped,
                              du_dpcs, dcov2d_dpcs, dpcs_dqcw)
        return image

    @staticmethod
    def backward(ctx, dloss_dgammas):
        cam = ctx.cam
        Rcw = ctx.Rcw
        
        us, cinv2ds, alphas, \
            depths, color_output, contrib, final_tau, \
            patch_range_per_tile, gsid_per_patch, \
            dcinv2d_dcov2ds, dcov2d_dscales, \
            du_dpcs, dcov2d_dpcs, dpcs_dqcw = ctx.saved_tensors

        dloss_dus, dloss_dcinv2ds, dloss_dalphas, dloss_dcolors =\
            gsc.splatB(cam.height, cam.width, us, cinv2ds, alphas,
                       depths, color_output, contrib, final_tau,
                       patch_range_per_tile, gsid_per_patch, dloss_dgammas)

        # splatB returns tensors with shape (N, 1, X), squeeze the middle dimension
        dloss_dus = dloss_dus.squeeze(1)  # (N, 2)
        dloss_dcinv2ds = dloss_dcinv2ds.squeeze(1)  # (N, 3)
        dloss_dalphas = dloss_dalphas.squeeze(1)  # (N, 1)
        dloss_dcolors = dloss_dcolors.squeeze(1)  # (N, 3)

        # Compute gradients w.r.t camera-space positions
        # dloss_dcinv2ds shape: (N, 3), dcinv2d_dcov2ds shape: (N, 3, 3)
        dloss_dcov2ds = torch.bmm(dloss_dcinv2ds.unsqueeze(1), dcinv2d_dcov2ds).squeeze(1)  # (N, 3)
        
        # dloss_dus shape: (N, 2), du_dpcs shape: (N, 2, 3)
        dloss_dpcs_from_us = torch.bmm(dloss_dus.unsqueeze(1), du_dpcs).squeeze(1)  # (N, 3)
        
        # dcov2d_dpcs shape: (N, 3, 3)
        dloss_dpcs_from_cov2ds = torch.bmm(dloss_dcov2ds.unsqueeze(1), dcov2d_dpcs).squeeze(1)  # (N, 3)
        
        dloss_dpcs = dloss_dpcs_from_us + dloss_dpcs_from_cov2ds

        # Camera parameter gradients
        # dloss_dpcs shape: (N, 3), dpcs_dqcw shape: (N, 3, 4)
        dloss_dqcw = torch.bmm(dloss_dpcs.unsqueeze(1), dpcs_dqcw).squeeze(1)  # (N, 4)
        dloss_dqcw = torch.sum(dloss_dqcw, dim=0)  # (4,)
        dloss_dtcw = torch.sum(dloss_dpcs, dim=0)  # (3,)

        # pws gradient - None (fixed)
        dloss_dpws = None

        # Colors gradient (simple RGB)
        dloss_dcolors_rgb = dloss_dcolors

        # Scales gradient (isotropic) - direct from cov2d
        # dloss_dcov2ds already computed above, dcov2d_dscales is (N, 3)
        # Element-wise multiply and sum since isotropic
        dloss_dscales = (dloss_dcov2ds * dcov2d_dscales).sum(dim=1)  # (N,)

        # Match input shapes: both alphas and scales are squeezed to (N,)
        return dloss_dqcw.squeeze(), \
            dloss_dtcw.squeeze(), \
            dloss_dpws, \
            dloss_dcolors_rgb, \
            dloss_dalphas.squeeze(), \
            dloss_dscales, \
            None
