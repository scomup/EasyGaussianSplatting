import torch
import gsplatcu as gsc


def logit(x):
    """
    inverse of sigmoid
    """
    return torch.log(x/(1-x))


class GSFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pws,
        shs,
        alphas,
        scales,
        rots,
        us,
        cam,
    ):
        # more detail view forward.pdf
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

        # Store the static parameters in the context
        ctx.cam = cam
        # Keep relevant tensors for backward
        ctx.save_for_backward(us, cinv2ds, alphas,
                              depths, colors, contrib, final_tau,
                              patch_range_per_tile, gsid_per_patch,
                              dcinv2d_dcov2ds, dcov2d_dcov3ds,
                              dcov3d_drots, dcov3d_dscales, dcolor_dshs,
                              du_dpcs, dcov2d_dpcs, dcolor_dpws)
        return image, areas

    @staticmethod
    def backward(ctx, dloss_dgammas, _):
        # Retrieve the saved tensors and static parameters
        cam = ctx.cam
        us, cinv2ds, alphas, \
            depths, colors, contrib, final_tau,\
            patch_range_per_tile, gsid_per_patch,\
            dcinv2d_dcov2ds, dcov2d_dcov3ds,\
            dcov3d_drots, dcov3d_dscales, dcolor_dshs,\
            du_dpcs, dcov2d_dpcs, dcolor_dpws = ctx.saved_tensors

        # more detail view backward.pdf

        # section.5
        dloss_dus, dloss_dcinv2ds, dloss_dalphas, dloss_dcolors =\
            gsc.splatB(cam.height, cam.width, us, cinv2ds, alphas,
                       depths, colors, contrib, final_tau,
                       patch_range_per_tile, gsid_per_patch, dloss_dgammas)

        dpc_dpws = cam.Rcw
        dloss_dcov2ds = dloss_dcinv2ds @ dcinv2d_dcov2ds
        # backward.pdf equation (3)
        dloss_drots = dloss_dcov2ds @ dcov2d_dcov3ds @ dcov3d_drots
        # backward.pdf equation (4)
        dloss_dscales = dloss_dcov2ds @ dcov2d_dcov3ds @ dcov3d_dscales
        # backward.pdf equation (5)
        dloss_dshs = (dloss_dcolors.permute(0, 2, 1) @
                      dcolor_dshs).permute(0, 2, 1).squeeze()
        # backward.pdf equation (7)
        dloss_dpws = dloss_dus @ du_dpcs @ dpc_dpws + \
            dloss_dcolors @ dcolor_dpws + \
            dloss_dcov2ds @ dcov2d_dpcs @ dpc_dpws
        print("max dloss_dus")
        print(torch.max(dloss_dus))
        return dloss_dpws.squeeze(),\
            dloss_dshs.squeeze(),\
            dloss_dalphas.squeeze().unsqueeze(1),\
            dloss_dscales.squeeze(),\
            dloss_drots.squeeze(),\
            dloss_dus.squeeze(),\
            None


class GSModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cunt = None
        self.grad_accum = None

    def forward(
            self,
            pws,
            shs,
            alphas_raw,
            scales_raw,
            rots_raw,
            us,
            cam):
        # Limit the value of alphas: 0 < alphas < 1
        alphas = torch.sigmoid(alphas_raw)
        # Limit the value of scales > 0
        scales = torch.exp(scales_raw)
        # Limit the value of rot, normal of rots is 1
        rots = torch.nn.functional.normalize(rots_raw)

        # apply GSfunction (forward)
        image, areas = GSFunction.apply(pws, shs, alphas, scales, rots, us, cam)

        return image, areas

    def update_density_info(self, dloss_dus, areas):
        with torch.no_grad():
            visible = areas[:, 0] * areas[:, 1] > 0
            grad = torch.norm(dloss_dus, dim=-1, keepdim=True)
            if self.cunt is None:
                self.grad_accum = grad
                self.cunt = visible.to(torch.int32)
            else:
                self.cunt += visible
                self.grad_accum += grad
            pass
