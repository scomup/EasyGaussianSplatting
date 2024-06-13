import torch
import gsplatcu as gsc
from gsplat.utils import *


def update_params(optimizer, gs_params, gs_params_new):
    for group in optimizer.param_groups:
        param_new = gs_params_new[group["name"]]
        state = optimizer.state.get(group['params'][0], None)
        if state is not None:
            state["exp_avg"] = torch.cat(
                (state["exp_avg"], torch.zeros_like(param_new)), dim=0)
            state["exp_avg_sq"] = torch.cat(
                (state["exp_avg_sq"], torch.zeros_like(param_new)), dim=0)
            del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter(
                torch.cat((group["params"][0], param_new), dim=0).requires_grad_(True))
            optimizer.state[group['params'][0]] = state
            gs_params[group["name"]] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(
                torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
            gs_params[group["name"]] = group["params"][0]


def prune_params(optimizer, gs_params, mask):
    for group in optimizer.param_groups:
        state = optimizer.state.get(group['params'][0], None)
        if state is not None:
            state["exp_avg"] = state["exp_avg"][mask]
            state["exp_avg_sq"] = state["exp_avg_sq"][mask]
            del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter(
                (group["params"][0][mask].requires_grad_(True)))
            optimizer.state[group['params'][0]] = state
            gs_params[group["name"]] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(
                group["params"][0][mask].requires_grad_(True))
            gs_params[group["name"]] = group["params"][0]


def render(pws, shs, alphas, scales, rots, cam):
    # more detail view forward.pdf
    # step1. Transform pw to camera frame,
    # and project it to iamge.
    us, pcs, depths = gsc.project(
        pws, cam.Rcw, cam.tcw, cam.fx, cam.fy, cam.cx, cam.cy, False)

    # step2. Calcuate the 3d Gaussian.
    cov3ds = gsc.computeCov3D(
        rots, scales, depths, False)[0]

    # step3. Calcuate the 2d Gaussian.
    cov2ds = gsc.computeCov2D(
        cov3ds, pcs, cam.Rcw, depths, cam.fx, cam.fy, cam.width, cam.height, False)[0]

    # step4. get color info
    colors = gsc.sh2Color(
        shs, pws, cam.twc, False)[0]

    # step5. Blend the 2d Gaussian to image
    cinv2ds, areas = gsc.inverseCov2D(
        cov2ds, depths, False)
    image, _, _, _, _ =\
        gsc.splat(cam.height, cam.width,
                  us, cinv2ds, alphas, depths, colors, areas)
    return image


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
        """
        if (torch.isnan(pws).any()):
            print("pws nan")
        if (torch.isnan(alphas).any()):
            print("alphas nan")
        if (torch.isnan(scales).any()):
            print("scales nan")
        if (torch.isnan(rots).any()):
            print("rots nan")
        if (torch.isnan(shs).any()):
            print("shs nan")
        """

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
        """
        if (torch.isnan(dloss_dpws).any()):
            print("dloss_dpws nan")
        if (torch.isnan(dloss_dus).any()):
            print("dloss_dus nan")
        if (torch.isnan(du_dpcs).any()):
            print("du_dpcs nan")
        if (torch.isnan(dpc_dpws).any()):
            print("dpc_dpws nan")
        if (torch.isnan(dloss_dcolors).any()):
            print("dloss_dcolors nan")
        if (torch.isnan(dcolor_dpws).any()):
            print("dcolor_dpws nan")
        if (torch.isnan(dloss_dcov2ds).any()):
            print("dloss_dcov2ds nan")
        if (torch.isnan(dcov2d_dpcs).any()):
            print("dcov2d_dpcs nan")
        """

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
        self.cam = None
        self.grad_threshold = 0.0002 / 500
        self.scale_threshold = 0.01 * 5.6
        self.alpha_threshold = 0.005
        self.big_threshold = 0.1 * 5.6
        self.reset_alpha_val = 0.01

    def forward(
            self,
            pws,
            shs,
            alphas_raw,
            scales_raw,
            rots_raw,
            us,
            cam):
        self.cam = cam
        # Limit the value of alphas: 0 < alphas < 1
        alphas = get_alphas(alphas_raw)
        # Limit the value of scales > 0
        scales = get_scales(scales_raw)
        # Limit the value of rot, normal of rots is 1
        rots = torch.nn.functional.normalize(rots_raw)

        # apply GSfunction (forward)
        image, areas = GSFunction.apply(pws, shs, alphas, scales, rots, us, cam)

        return image, areas

    def update_density_info(self, dloss_dus, areas):
        with torch.no_grad():
            visible = areas[:, 0] > 0
            dloss_dus[:, 0] = dloss_dus[:, 0]
            dloss_dus[:, 0] = dloss_dus[:, 1]
            grad = torch.norm(dloss_dus, dim=-1, keepdim=True)

            if self.cunt is None:
                self.grad_accum = grad
                self.cunt = visible.to(torch.int32)
            else:
                self.cunt += visible
                self.grad_accum[visible] += grad[visible]
            pass

    def update_gaussian_density(self, gs_params, optimizer):
        # prune too small or too big gaussian
        selected_by_small_alpha = gs_params["alphas_raw"].squeeze() < get_alphas_raw(self.alpha_threshold)
        selected_by_big_scale = torch.max(gs_params["scales_raw"], axis=1)[0] > get_scales_raw(self.big_threshold)
        selected_for_prune = torch.logical_or(selected_by_small_alpha, selected_by_big_scale)
        selected_for_remain = torch.logical_not(selected_for_prune)
        prune_params(optimizer, gs_params, selected_for_remain)

        grads = self.grad_accum.squeeze()[selected_for_remain] / self.cunt[selected_for_remain]
        grads[grads.isnan()] = 0.0

        pws = gs_params["pws"]
        shs = gs_params["shs"]
        alphas = get_alphas(gs_params["alphas_raw"])
        scales = get_scales(gs_params["scales_raw"])
        rots = get_rots(gs_params["rots_raw"])

        selected_by_grad = grads >= self.grad_threshold
        selected_by_scale = torch.max(scales, axis=1)[0] <= self.scale_threshold

        selected_for_clone = torch.logical_and(selected_by_grad, selected_by_scale)
        selected_for_split = torch.logical_and(selected_by_grad, torch.logical_not(selected_by_scale))

        # clone gaussians
        pws_cloned = pws[selected_for_clone]
        shs_cloned = shs[selected_for_clone]
        alphas_cloned = alphas[selected_for_clone]
        scales_cloned = scales[selected_for_clone]
        rots_cloned = rots[selected_for_clone]

        # split gaussians
        # try:
        #     Cov3d = compute_cov_3d_torch(
        #         scales[selected_for_split], rots[selected_for_split])
        #     multi_normal = torch.distributions.MultivariateNormal(
        #         loc=pws[selected_for_split], covariance_matrix=Cov3d)
        #     pws_splited = multi_normal.sample()  # sampling new pw for splited gaussian
        # except Exception as e:
        #     print(e)

        rots_splited = rots[selected_for_split]
        means = torch.zeros((rots_splited.size(0), 3), device="cuda")
        stds = scales[selected_for_split]
        samples = torch.normal(mean=means, std=stds)
        # sampling new pw for splited gaussian
        pws_splited = pws[selected_for_split] + \
            rotate_vector_by_quaternion(rots_splited, samples)
        alphas_splited = alphas[selected_for_split]
        scales[selected_for_split] = scales[selected_for_split] * 0.6  # splited gaussian will go smaller
        scales_splited = scales[selected_for_split]
        shs_splited = shs[selected_for_split]

        gs_params_new = {"pws": torch.cat([pws_cloned, pws_splited]),
                         "shs": torch.cat([shs_cloned, shs_splited]),
                         "alphas_raw": get_alphas_raw(torch.cat([alphas_cloned, alphas_splited])),
                         "scales_raw": get_scales_raw(torch.cat([scales_cloned, scales_splited])),
                         "rots_raw": torch.cat([rots_cloned, rots_splited])}

        debug = True
        if (debug):
            rgb = torch.Tensor([1, 0, 0]).to(torch.float32).to('cuda')
            flat_shs = (rgb - 0.5) / 0.28209479177387814
            flat_shs = flat_shs.repeat(gs_params_new['pws'].shape[0], 1)
            update_params(optimizer, gs_params, gs_params_new)
            gs_params_new['shs'] = flat_shs
            debug_gs = {"pws": torch.cat([gs_params["pws"], gs_params_new["pws"]]),
                        "shs": torch.cat([gs_params["pws"], flat_shs]),
                        "alphas_raw": torch.cat([gs_params["alphas_raw"], gs_params_new["alphas_raw"]]),
                        "scales_raw": torch.cat([gs_params["scales_raw"], gs_params_new["scales_raw"]]),
                        "rots_raw": torch.cat([gs_params["rots_raw"], gs_params_new["rots_raw"]])}

        # split gaussians (N is the split size)
        update_params(optimizer, gs_params, gs_params_new)

        print("gaussian density update report")
        prune_n = int(torch.sum(selected_for_prune))
        clone_n = int(torch.sum(selected_for_clone))
        split_n = int(torch.sum(selected_for_split))
        print("pruned num: ", prune_n)
        print("cloned num: ", clone_n)
        print("splited num: ", split_n)
        print("gaussian num change: ", clone_n + split_n - prune_n)

        self.grad_accum = None
        self.cunt = None

        if (debug):
            return debug_gs


    def reset_alpha(self, gs_params, optimizer):
        reset_alpha_raw_val = get_alphas_raw(self.reset_alpha_val)
        rest_mask = gs_params['alphas_raw'] > reset_alpha_raw_val
        gs_params['alphas_raw'][rest_mask] = torch.ones_like(gs_params['alphas_raw'])[rest_mask] * reset_alpha_raw_val

        alpha_param = list(filter(lambda x: x["name"] == "alphas_raw", optimizer.param_groups))[0]

        state = optimizer.state.get(alpha_param['params'][0], None)
        state["exp_avg"] = torch.zeros_like(gs_params['alphas_raw'])
        state["exp_avg_sq"] = torch.zeros_like(gs_params['alphas_raw'])
        del optimizer.state[alpha_param['params'][0]]
        optimizer.state[alpha_param['params'][0]] = state
