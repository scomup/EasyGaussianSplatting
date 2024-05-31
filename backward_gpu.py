import matplotlib.pyplot as plt
import torch
import gsplatcu as gsc
import numpy as np
from gsplat.sh_coef import *
from backward_cpu import *
from gsplat.gau_io import *


if __name__ == "__main__":
    sh_rest_dim = 45
    sh_dim = 3 + sh_rest_dim
    gs = get_example_gs()
    gs_num = gs.shape[0]
    rest_shs = np.random.rand(gs_num, sh_rest_dim)
    shs = np.concatenate((gs['sh'], rest_shs), axis=1).astype(np.float64)

    # Camera info
    tcw = np.array([1.03796196, 0.42017467, 4.67804612])
    Rcw = np.array([[0.89699204,  0.06525223,  0.43720409],
                    [-0.04508268,  0.99739184, -0.05636552],
                    [-0.43974177,  0.03084909,  0.89759429]]).T
    twc = np.linalg.inv(Rcw) @ (-tcw)
    width = int(32)
    height = int(16)
    fx = 16
    fy = 16
    cx = width/2.
    cy = height/2.
    image_gt = np.zeros([height, width, 3])

    pws = gs['pw'].astype(np.float64)
    alphas = gs['alpha'].astype(np.float64)
    rots = gs['rot'].astype(np.float64)
    scales = gs['scale'].astype(np.float64)

    colors = np.zeros([gs_num, 3])
    us = np.zeros([gs_num, 2])
    pcs = np.zeros([gs_num, 3])
    cov3ds = np.zeros([gs_num, 6])
    cov2ds = np.zeros([gs_num, 3])
    cinv2ds = np.zeros([gs_num, 3])
    dpc_dpws = np.zeros([gs_num, 3, 3])
    du_dpcs = np.zeros([gs_num, 2, 3])
    dcov3d_drots = np.zeros([gs_num, 6, 4])
    dcov3d_dscales = np.zeros([gs_num, 6, 3])
    dcov2d_dcov3ds = np.zeros([gs_num, 3, 6])
    dcov2d_dpcs = np.zeros([gs_num, 3, 3])
    dcolor_dshs = np.zeros([gs_num, 1, sh_dim//3])
    dcolor_dpws = np.zeros([gs_num, 3, 3])
    dcinv2d_dcov2ds = np.zeros([gs_num, 3, 3])
    for i in range(gs_num):
        pcs[i], dpc_dpws[i] = transform(pws[i], Rcw, tcw, True)
        us[i], du_dpcs[i] = project(pcs[i], fx, fy, cx, cy, True)
        cov3ds[i], dcov3d_drots[i], dcov3d_dscales[i] = compute_cov_3d(
            rots[i], scales[i], True)
        cov2ds[i], dcov2d_dcov3ds[i], dcov2d_dpcs[i] = compute_cov_2d(
            cov3ds[i], pcs[i], Rcw, fx, fy, True)
        colors[i], dcolor_dshs[i], dcolor_dpws[i] = sh2color(
            shs[i], pws[i], twc, True)
        cinv2ds[i], dcinv2d_dcov2ds[i] = calc_cinv2d(cov2ds[i], True)

    image = get_image(alphas, cinv2ds, colors, us, height, width)

    loss, dloss_dalphas, dloss_dcinv2ds, dloss_dcolors, dloss_dus = calc_loss(
        alphas, cinv2ds, colors, us, image_gt, True)
    dloss_dalphas = dloss_dalphas.reshape([gs_num, 1, 1])
    dloss_dcinv2ds = dloss_dcinv2ds.reshape([gs_num, 1, 3])
    dloss_dcolors = dloss_dcolors.reshape([gs_num, 1, 3])
    dloss_dus = dloss_dus.reshape([gs_num, 1, 2])

    pws_gpu = torch.from_numpy(pws).type(torch.float32).to('cuda')
    rots_gpu = torch.from_numpy(rots).type(torch.float32).to('cuda')
    scales_gpu = torch.from_numpy(scales).type(torch.float32).to('cuda')
    alphas_gpu = torch.from_numpy(alphas).type(torch.float32).to('cuda')
    shs_gpu = torch.from_numpy(shs).type(torch.float32).to('cuda')
    Rcw_gpu = torch.from_numpy(Rcw).type(torch.float32).to('cuda')
    tcw_gpu = torch.from_numpy(tcw).type(torch.float32).to('cuda')
    twc_gpu = torch.from_numpy(twc).type(torch.float32).to('cuda')

    us_gpu, pcs_gpu, _, du_dpcs_gpu = gsc.project(
        pws_gpu, Rcw_gpu, tcw_gpu, fx, fy, cx, cy, True)
    print("%s test us_gpu" % check(us_gpu.cpu().numpy(), us))
    print("%s test pcs_gpu" % check(pcs_gpu.cpu().numpy(), pcs))
    print("%s test du_dpcs_gpu" % check(du_dpcs_gpu.cpu().numpy(), du_dpcs))

    depths_gpu = torch.from_numpy(
        np.array([1, 2, 3, 4])).type(torch.float32).to('cuda')

    cov3ds_gpu, dcov3d_drots_gpu, dcov3d_dscales_gpu = gsc.computeCov3D(
        rots_gpu, scales_gpu, depths_gpu, True)
    print("%s test cov3ds_gpu" % check(cov3ds_gpu.cpu().numpy(), cov3ds))
    print("%s test dcov3d_drots_gpu" %
          check(dcov3d_drots_gpu.cpu().numpy(), dcov3d_drots))
    print("%s test dcov3d_dscales_gpu" %
          check(dcov3d_dscales_gpu.cpu().numpy(), dcov3d_dscales))

    cov2ds_gpu, dcov2d_dcov3ds_gpu, dcov2d_dpcs_gpu = gsc.computeCov2D(
        cov3ds_gpu, pcs_gpu, Rcw_gpu, depths_gpu, fx, fy, width, height, True)
    print("%s test cov2ds_gpu" % check(cov2ds_gpu.cpu().numpy(), cov2ds))
    print("%s test dcov2d_dcov3ds_gpu" %
          check(dcov2d_dcov3ds_gpu.cpu().numpy(), dcov2d_dcov3ds))
    print("%s test dcov2d_dpcs_gpu" %
          check(dcov2d_dpcs_gpu.cpu().numpy(), dcov2d_dpcs))

    colors_gpu, dcolor_dshs_gpu, dcolor_dpws_gpu = gsc.sh2Color(
        shs_gpu, pws_gpu, twc_gpu, True)
    print("%s test colors_gpu" % check(colors_gpu.cpu().numpy(), colors))
    print("%s test dcolor_dshs_gpu" %
          check(dcolor_dshs_gpu.cpu().numpy(), dcolor_dshs))
    print("%s test dcolor_dpws_gpu" %
          check(dcolor_dpws_gpu.cpu().numpy(), dcolor_dpws))

    cinv2ds_gpu, areas_gpu, dcinv2d_dcov2ds_gpu = gsc.inverseCov2D(
        cov2ds_gpu, depths_gpu, True)
    print("%s test cinv2d_gpu" % check(cinv2ds_gpu.cpu().numpy(), cinv2ds))
    print("%s test dcinv2d_dcov2ds_gpu" %
          check(dcinv2d_dcov2ds_gpu.cpu().numpy(), dcinv2d_dcov2ds))

    image_gpu, contrib_gpu, final_tau_gpu, patch_range_per_tile_gpu, gsid_per_patch_gpu =\
        gsc.splat(height, width, us_gpu, cinv2ds_gpu,
                  alphas_gpu, depths_gpu, colors_gpu, areas_gpu)
    print("%s test image_gpu" %
          check(image_gpu.cpu().numpy().transpose([1, 2, 0]), image))

    # f, axarr = plt.subplots(1,2)
    # axarr[0].imshow(image)
    # axarr[1].imshow(image_gpu.cpu().numpy().transpose([1, 2, 0]))
    # plt.show()
    # exit()

    _, dloss_dgammas = get_loss(image, image_gt)
    dloss_dgammas_gpu = torch.from_numpy(
        dloss_dgammas).type(torch.float32).to('cuda')

    dloss_dus_gpu, dloss_dcinv2ds_gpu, dloss_dalphas_gpu, dloss_dcolors_gpu =\
        gsc.splatB(height, width, us_gpu, cinv2ds_gpu, alphas_gpu, depths_gpu, colors_gpu,
                   contrib_gpu, final_tau_gpu, patch_range_per_tile_gpu, gsid_per_patch_gpu, dloss_dgammas_gpu)

    dloss_dalphas = dloss_dalphas.reshape([gs_num, 1, 1])
    dloss_dcinv2ds = dloss_dcinv2ds.reshape([gs_num, 1, 3])
    dloss_dcolors = dloss_dcolors.reshape([gs_num, 1, 3])
    dloss_dus = dloss_dus.reshape([gs_num, 1, 2])

    print("%s test dloss_dus_gpu" %
          check(dloss_dus_gpu.cpu().numpy(), dloss_dus))
    print("%s test dloss_dcinv2ds_gpu" %
          check(dloss_dcinv2ds_gpu.cpu().numpy(), dloss_dcinv2ds))
    print("%s test dloss_dalphas_gpu" %
          check(dloss_dalphas_gpu.cpu().numpy(), dloss_dalphas))
    print("%s test dloss_dcolors_gpu" %
          check(dloss_dcolors_gpu.cpu().numpy(), dloss_dcolors))
    dpc_dpws_gpu = Rcw_gpu

    dloss_drots_gpu = dloss_dcinv2ds_gpu @ dcinv2d_dcov2ds_gpu @ dcov2d_dcov3ds_gpu @ dcov3d_drots_gpu
    dloss_dscales_gpu = dloss_dcinv2ds_gpu @ dcinv2d_dcov2ds_gpu @ dcov2d_dcov3ds_gpu @ dcov3d_dscales_gpu
    dloss_dshs_gpu = (dloss_dcolors_gpu.permute(0, 2, 1) @
                      dcolor_dshs_gpu).permute(0, 2, 1).reshape(gs_num, 1, -1)
    dloss_dalphas_gpu = dloss_dalphas_gpu
    dloss_dpws_gpu = dloss_dus_gpu @ du_dpcs_gpu @ dpc_dpws_gpu + \
        dloss_dcolors_gpu @ dcolor_dpws_gpu + \
        dloss_dcinv2ds_gpu @ dcinv2d_dcov2ds_gpu @ dcov2d_dpcs_gpu @ dpc_dpws_gpu
    pass
