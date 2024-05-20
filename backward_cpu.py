import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from sh_coef import *


def upper_triangular(mat):
    s = mat.shape[0]
    n = 0
    if (s == 2):
        n = 3
    elif(s == 3):
        n = 6
    else:
        raise NotImplementedError("no supported mat")
    upper = np.zeros([n])
    n = 0
    for i in range(s):
        for j in range(i, s):
            upper[n] = mat[i, j]
            n = n + 1
    return upper


def symmetric_matrix(upper):
    n = upper.shape[0]
    if (n == 6):
        s = 3
    elif(n == 3):
        s = 2
    else:
        raise NotImplementedError("no supported mat")
    mat = np.zeros([s, s])

    n = 0
    for i in range(s):
        for j in range(i, s):
            mat[i, j] = upper[n]
            if (i != j):
                mat[j, i] = upper[n]
            n = n + 1
    return mat


def numerical_derivative(func, param, idx, plus=lambda a, b: a + b, minus=lambda a, b: a - b, delta=1e-8):
    r = func(*param)
    m = r.shape[0]
    n = param[idx].shape[0]
    J = np.zeros([m, n])
    for j in range(n):
        dx = np.zeros(n)
        dx[j] = delta
        param_delta = param.copy()
        param_delta[idx] = plus(param[idx], dx)
        J[:, j] = minus(func(*param_delta), r)/delta
    return J


def check(a, b):
    return np.all(np.abs(a - b) < 0.0001)


def transform(pw, Rcw, tcw, calc_J=False):
    pc = Rcw @ pw + tcw
    if (calc_J):
        dpc_dpw = Rcw
        return pc, dpc_dpw
    else:
        return pc


def project(pc, fx, fy, cx, cy, calc_J=False):
    x, y, z = pc
    z_2 = z * z
    u = np.array([(x * fx / z + cx),
                  (y * fy / z + cy)])
    if (calc_J is True):
        dpc_du = np.array([[fx / z,    0, -fx * x / z_2],
                           [0, fy / z, -fy * y / z_2]])
        return u, dpc_du
    else:
        return u


def calc_m(q, s, calc_J=False):
    w, x, y, z = q
    s0, s1, s2 = s
    R = np.array([
        [1.0 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x * z + y * w)],
        [2*(x*y + z*w), 1.0 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1.0 - 2*(x**2 + y**2)]
    ])
    S = np.diag(s)
    M = R @ S
    m = M.reshape(-1)
    if (calc_J):
        dm_rot = np.array([[0,        0,      -4*s0*y,  -4*s0*z],
                           [-2*s1*z,  2*s1*y,  2*s1*x,  -2*s1*w],
                           [2*s2*y,   2*s2*z,  2*s2*w,  2*s2*x],
                           [2*s0*z,   2*s0*y,  2*s0*x,  2*s0*w],
                           [0,       -4*s1*x,  0,      -4*s1*z],
                           [-2*s2*x, -2*s2*w,  2*s2*z,  2*s2*y],
                           [-2*s0*y,  2*s0*z, -2*s0*w,  2*s0*x],
                           [2*s1*x,   2*s1*w,  2*s1*z,  2*s1*y],
                           [0,       -4*s2*x, -4*s2*y,  0]])
        dm_s = np.zeros([9, 3])
        dm_s[0:3, :] = np.diag(R[0])
        dm_s[3:6, :] = np.diag(R[1])
        dm_s[6:9, :] = np.diag(R[2])
        return m, dm_rot, dm_s
    else:
        return m


def calc_mmt(m, calc_J=False):
    M = m.reshape([3, 3])
    MMT = M @ M.T
    mmt = np.array([MMT[0, 0], MMT[0, 1], MMT[0, 2], MMT[1, 1], MMT[1, 2], MMT[2, 2]])
    if (calc_J):
        # a, b, c, d, e, f, g, h, i = m
        # |a b c| |a d g|   |aa+bb+cc  ad+be+cg  ag+bh+ci|
        # |d e f| |b e h| = |          dd+ee+ff  dg+eh+fi|
        # |g h i| |c f i|   |                    gg+hh+ii|
        dmmt_m = np.zeros([6, 9])
        dmmt_m[0, 0:3] = M[0]*2
        dmmt_m[1, 0:3] = M[1]
        dmmt_m[1, 3:6] = M[0]
        dmmt_m[2, 0:3] = M[2]
        dmmt_m[2, 6:9] = M[0]
        dmmt_m[3, 3:6] = M[1]*2
        dmmt_m[4, 3:6] = M[2]
        dmmt_m[4, 6:9] = M[1]
        dmmt_m[5, 6:9] = M[2]*2
        return mmt, dmmt_m
    else:
        return mmt


def compute_cov_3d(q, s, calc_J=False):
    m, dm_mq, dm_ds = calc_m(q, s, True)
    cov, dcov_dm = calc_mmt(m, True)
    if (calc_J):
        return cov, dcov_dm @ dm_mq, dcov_dm @ dm_ds
    else:
        return cov


def compute_cov_2d(cov3d, pc, Rcw, fx, fy, calc_J=False):
    Cov3d = symmetric_matrix(cov3d)
    J = get_J(pc, fx, fy)
    M = J @ Rcw
    Cov2d = M @ Cov3d @ M.T
    cov2d = upper_triangular(Cov2d)
    if (calc_J):
        m00, m01, m02 = M[0]
        m10, m11, m12 = M[1]
        a, b, c, d, e, f = cov3d
        x, y, z = pc
        r00, r01, r02, r10, r11, r12, r20, r21, r22 = Rcw.reshape(-1)

        dcov2d_dcov3d =\
            np.array([[m00**2, 2*m00*m01, 2*m00*m02, m01**2, 2*m01*m02, m02**2],
                      [m00*m10, m00*m11 + m01*m10, m00*m12 + m02 * m10, m01*m11, m01*m12 + m02*m11, m02*m12],
                      [m10**2, 2*m10*m11, 2*m10*m12, m11**2, 2*m11*m12, m12**2]])

        dcov2d_dm =\
            np.array([[2*a*m00 + 2*b*m01 + 2*c*m02, 2*b*m00 + 2*d*m01 + 2*e*m02, 2*c*m00 + 2*e*m01 + 2*f*m02, 0, 0, 0],
                      [a*m10 + b*m11 + c*m12, b*m10 + d*m11 + e*m12, c*m10 + e*m11 + f*m12,
                          a*m00 + b*m01 + c*m02, b*m00 + d*m01 + e*m02, c*m00 + e*m01 + f*m02],
                      [0, 0, 0, 2*a*m10 + 2*b*m11 + 2*c*m12, 2*b*m10 + 2*d*m11 + 2*e*m12, 2*c*m10 + 2*e*m11 + 2*f*m12]])
        dM_dpc =\
            np.array([[-fx*r00/x**2 - fx*r20/z**2, 0, 2*fx*r20*x/z**3],
                      [-fx*r01/x**2 - fx*r21/z**2, 0, 2*fx*r21*x/z**3],
                      [-fx*r02/x**2 - fx*r22/z**2, 0, 2*fx*r22*x/z**3],
                      [0, -fy*r20/z**2, -fy*r10/z**2 + 2*fy*r20*y/z**3],
                      [0, -fy*r21/z**2, -fy*r11/z**2 + 2*fy*r21*y/z**3],
                      [0, -fy*r22/z**2, -fy*r12/z**2 + 2*fy*r22*y/z**3]])

        return cov2d, dcov2d_dcov3d, dcov2d_dm @ dM_dpc
    else:
        return cov2d


def get_J(pc, fx, fy):
    x, y, z = pc
    z2 = z*z
    return np.array([[fx/x, 0, -fx*x/z2],
                     [0, fy/z, -fy*y/z2]])


def calc_cinv2d(cov2d, calc_J=False):
    det_inv = 1. / (cov2d[0] * cov2d[2] - cov2d[1] * cov2d[1])
    cinv2d = np.array([cov2d[2], -cov2d[1], cov2d[0]]) * det_inv
    if (calc_J):
        a, b, c = cov2d
        det_inv2 = det_inv * det_inv
        J = np.array([[-c*c*det_inv2, 2*b*c*det_inv2, -a*c*det_inv2 + det_inv],
                      [b*c*det_inv2, -2*b*b*det_inv2 - det_inv, a*b*det_inv2],
                      [-a*c*det_inv2 + det_inv, 2*a*b*det_inv2, -a*a*det_inv2]])
        return cinv2d, J

    else:
        return cinv2d


def calc_alpha_prime(alpha, cinv2d, u, x, calc_J=False):
    d = u - x
    maha_dist = cinv2d[0] * d[0] * d[0] + cinv2d[2] * \
        d[1] * d[1] + 2 * cinv2d[1] * d[0] * d[1]
    g = np.exp(-0.5 * maha_dist)
    alphaprime = g * alpha
    if (calc_J):
        dalphaprime_dalpha = np.array([[g]])
        dalphaprime_dcinv2d = -0.5 * alphaprime * \
            np.array([d[0] * d[0], 2 * d[0] * d[1], d[1] * d[1]])
        dalphaprime_du = alphaprime * \
            np.array([-cinv2d[0]*d[0] - cinv2d[1]*d[1],
                     (-cinv2d[1]*d[0] - cinv2d[2]*d[1])])
        return alphaprime, dalphaprime_dalpha, dalphaprime_dcinv2d.reshape([1, 3]), dalphaprime_du.reshape([1, 2])
    else:
        return alphaprime


def calc_gamma(alphas, cov2ds, colors, us, x, calc_J=False):
    cont_tmp = 0
    cont = 0
    cov2ds = cov2ds.reshape([-1, 3])
    colors = colors.reshape([-1, 3])
    us = us.reshape([-1, 2])
    tau = 1.
    gamma = np.zeros(3)
    for alpha, cov2d, color, u in zip(alphas, cov2ds, colors, us):
        cont_tmp = cont_tmp + 1
        cinv2d = calc_cinv2d(cov2d)
        alpha_prime = calc_alpha_prime(alpha, cinv2d, u, x)
        if (alpha_prime < 0.002):
            continue
        cont = cont_tmp
        gamma += alpha_prime * color * tau
        tau = tau * (1 - alpha_prime)
        if (tau < 0.0001):
            break
    if (calc_J):
        gs_num = alphas.shape[0]
        gamma_cur2last = np.zeros(3)
        dgamma_dalpha = np.zeros([gs_num, 3, 1])
        dgamma_dcov2d = np.zeros([gs_num, 3, 3])
        dgamma_dcolor = np.zeros([gs_num, 3, 3])
        dgamma_du = np.zeros([gs_num, 3, 2])
        for i in reversed(range(cont)):
            alpha, cov2d, color, u = alphas[i], cov2ds[i], colors[i], us[i]
            cinv2d, dcinv2d_dcov2d = calc_cinv2d(cov2d, True)
            alpha_prime, dalphaprime_dalpha, dalphaprime_dcinv2d, dalphaprime_du =\
                calc_alpha_prime(alpha, cinv2d, u, x, True)
            if (alpha_prime < 0.002):
                continue
            tau = tau / (1 - alpha_prime)
            dgamma_dalphaprime = (
                tau * (color - gamma_cur2last)).reshape([3, 1])
            dgamma_dalpha[i] = dgamma_dalphaprime @ dalphaprime_dalpha
            dgamma_dcov2d[i] = dgamma_dalphaprime @ dalphaprime_dcinv2d @ dcinv2d_dcov2d
            dgamma_dcolor[i] = tau * alpha_prime * np.eye(3)
            dgamma_du[i] = dgamma_dalphaprime @ dalphaprime_du
            gamma_cur2last = alpha_prime * color + \
                (1 - alpha_prime) * gamma_cur2last
        return gamma, dgamma_dalpha, dgamma_dcov2d, dgamma_dcolor, dgamma_du, cont
    else:
        return gamma


def sh2color(sh, pw, twc, calc_J=False):
    sh_dim = sh.shape[0]
    dc_dsh = np.zeros([sh.shape[0]//3])
    dc_dpw = np.zeros([3, 3])
    dc_dsh[0] = SH_C0_0
    sh = sh.reshape([-1, 3])
    color = dc_dsh[0] * sh[0] + 0.5
    if (sh_dim > 3):
        d = pw - twc
        normd = np.linalg.norm(d)
        r = d / normd
        x, y, z = r

        dc_dsh[1] = SH_C1_0 * y
        dc_dsh[2] = SH_C1_1 * z
        dc_dsh[3] = SH_C1_2 * x
        color = color + \
            dc_dsh[1] * sh[1] + \
            dc_dsh[2] * sh[2] + \
            dc_dsh[3] * sh[3]

        if (sh_dim > 12):
            xx = x * x
            yy = y * y
            zz = z * z
            xy = x * y
            yz = y * z
            xz = x * z
            dc_dsh[4] = SH_C2_0 * xy
            dc_dsh[5] = SH_C2_1 * yz
            dc_dsh[6] = SH_C2_2 * (2.0 * zz - xx - yy)
            dc_dsh[7] = SH_C2_3 * xz
            dc_dsh[8] = SH_C2_4 * (xx - yy)

            color = color + \
                dc_dsh[4] * sh[4] + \
                dc_dsh[5] * sh[5] + \
                dc_dsh[6] * sh[6] + \
                dc_dsh[7] * sh[7] + \
                dc_dsh[8] * sh[8]

            if (sh_dim > 27):
                dc_dsh[9] = SH_C3_0 * y * (3.0 * xx - yy)
                dc_dsh[10] = SH_C3_1 * xy * z
                dc_dsh[11] = SH_C3_2 * y * (4.0 * zz - xx - yy)
                dc_dsh[12] = SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)
                dc_dsh[13] = SH_C3_4 * x * (4.0 * zz - xx - yy)
                dc_dsh[14] = SH_C3_5 * z * (xx - yy)
                dc_dsh[15] = SH_C3_6 * x * (xx - 3.0 * yy)

                color = color +  \
                    dc_dsh[9] * sh[9] + \
                    dc_dsh[10] * sh[10] + \
                    dc_dsh[11] * sh[11] + \
                    dc_dsh[12] * sh[12] + \
                    dc_dsh[13] * sh[13] + \
                    dc_dsh[14] * sh[14] + \
                    dc_dsh[15] * sh[15]
    if (calc_J):
        dc_dpc = np.zeros([3, 3])
        if (sh_dim > 3):
            dr_dpc = np.zeros([3, 3])
            normd3_inv = 1/normd**3
            normd_inv = 1/normd
            dr_dpc[0, 0] = -d[0]*d[0]*normd3_inv + normd_inv
            dr_dpc[1, 1] = -d[1]*d[1]*normd3_inv + normd_inv
            dr_dpc[2, 2] = -d[2]*d[2]*normd3_inv + normd_inv
            dr_dpc[0, 1] = -d[0]*d[1]*normd3_inv
            dr_dpc[0, 2] = -d[0]*d[2]*normd3_inv
            dr_dpc[1, 2] = -d[1]*d[2]*normd3_inv
            dr_dpc[0, 1] = dr_dpc[1, 0]
            dr_dpc[0, 2] = dr_dpc[2, 0]
            dr_dpc[2, 1] = dr_dpc[1, 2]

            dc_dr = np.zeros([3, 3])
            dc_dr[:, 0] += SH_C1_2 * sh[3]
            dc_dr[:, 1] += SH_C1_0 * sh[1]
            dc_dr[:, 2] += SH_C1_1 * sh[2]
            if (sh_dim > 12):
                xx = x * x
                yy = y * y
                zz = z * z
                xy = x * y
                yz = y * z
                xz = x * z
                dc_dr[:, 0] += SH_C2_0 * y * sh[4] - SH_C2_2 * 2 * \
                    x * sh[6] + SH_C2_3 * z * sh[7] + SH_C2_4 * 2 * x * sh[8]
                dc_dr[:, 1] += SH_C2_0 * x * sh[4] + SH_C2_1 * z * sh[5] - \
                    SH_C2_2 * 2.0 * y * sh[6] - SH_C2_4 * 2 * y * sh[8]
                dc_dr[:, 2] += SH_C2_1 * y * sh[5] + SH_C2_2 * \
                    (4.0 * z) * sh[6] + SH_C2_3 * x * sh[7]
                print(dc_dr @ dr_dpc)

        return color, dc_dsh, dc_dr @ dr_dpc
    else:
        return color


sh = np.arange(27)
# sh = np.array([1, 2, 3])
pw = np.array([0.5, 0.6, -1.0])
twc = np.array([0.23, -0.4, 0.45])
dc_dpw0 = numerical_derivative(sh2color, [sh, pw, twc], 1)
print(dc_dpw0)
_, _, dc_dr = sh2color(sh, pw, twc, True)
print("diff:", dc_dpw0 - dc_dr)
exit(0)


def calc_loss(alphas, cov2ds, colors, us, width, height, calc_J=False):
    image = np.zeros([height, width, 3])
    xs = np.indices([width, height]).reshape(2, -1).T
    for x in xs:
        gamma = calc_gamma(alphas, cov2ds, colors, us, x)
        image[x[1], x[0]] = gamma
    criterion = nn.L1Loss()
    image_tensor = torch.tensor(image.transpose([2, 0, 1]), requires_grad=True)
    image_gt = torch.zeros([3, height, width], dtype=torch.double)
    loss = criterion(image_tensor, image_gt)
    loss_val = loss.detach().numpy().reshape(1)
    if (calc_J):
        contrib = np.ones([height, width])
        loss.backward()
        dloss_dgammas = image_tensor.grad.detach().numpy()
        gs_num = alphas.shape[0]
        dloss_dalphas = np.zeros([gs_num, 1])
        dloss_dcov2ds = np.zeros([gs_num, 3])
        dloss_dcolors = np.zeros([gs_num, 3])
        dloss_dus = np.zeros([gs_num, 2])
        for x in xs:
            gamma, dgamma_dalphas, dgamma_dcov2ds, dgamma_dcolors, dgamma_dus, cont =\
                calc_gamma(alphas, cov2ds, colors, us, x, True)
            dloss_dgamma = dloss_dgammas[:, x[1], x[0]]
            contrib[x[1], x[0]] = cont
            for i in range(cont):
                dloss_dalphas[i] += dloss_dgamma @ dgamma_dalphas[i]
                dloss_dcov2ds[i] += dloss_dgamma @ dgamma_dcov2ds[i]
                dloss_dcolors[i] += dloss_dgamma @ dgamma_dcolors[i]
                dloss_dus[i] += dloss_dgamma @ dgamma_dus[i]
        return loss_val, \
            dloss_dalphas.reshape(1, -1), \
            dloss_dcov2ds.reshape(1, -1), \
            dloss_dcolors.reshape(1, -1), \
            dloss_dus.reshape(1, -1)
    else:
        return loss_val


if __name__ == "__main__":
    gs_data = np.random.rand(4, 59)
    gs_data0 = np.array([[0.,  0.,  0.,  # xyz
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
                         ], dtype=np.float64)

    gs_data[:, :14] = gs_data0
    dtypes = [('pos', '<f8', (3,)),
              ('rot', '<f8', (4,)),
              ('scale', '<f8', (3,)),
              ('alpha', '<f8'),
              ('sh', '<f8', (48,))]
    gs = np.frombuffer(gs_data.tobytes(), dtype=dtypes)
    gs_num = gs.shape[0]

    # Camera info
    tcw = np.array([1.03796196, 0.42017467, 4.67804612])
    Rcw = np.array([[0.89699204,  0.06525223,  0.43720409],
                    [-0.04508268,  0.99739184, -0.05636552],
                    [-0.43974177,  0.03084909,  0.89759429]]).T

    width = int(32)  # 1957  # 979
    height = int(16)  # 1091  # 546
    fx = 16
    fy = 16
    cx = width/2.
    cy = height/2.
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1.]])

    Tcw = np.eye(4)
    Tcw[:3, :3] = Rcw
    Tcw[:3, 3] = tcw
    cam_center = np.linalg.inv(Tcw)[:3, 3]

    pws = gs['pos']
    gs_num = gs['pos'].shape[0]

    colors = np.zeros([gs_num, 3])
    us = np.zeros([gs_num, 2])
    pcs = np.zeros([gs_num, 3])
    cov3ds = np.zeros([gs_num, 6])
    cov2ds = np.zeros([gs_num, 3])
    dpc_dpws = np.zeros([gs_num, 3, 3])
    du_dpcs = np.zeros([gs_num, 2, 3])
    dcov3d_drots = np.zeros([gs_num, 6, 4])
    dcov3d_dscales = np.zeros([gs_num, 6, 3])
    dcov2d_dcov3ds = np.zeros([gs_num, 3, 6])
    dcov2d_dpcs = np.zeros([gs_num, 3, 3])
    dcolor_dshs = np.zeros([gs_num, gs['sh'].shape[1]//3])
    for i in range(gs_num):
        # step1. Transform pw to camera frame,
        # and project it to iamge.
        pcs[i], dpc_dpws[i] = transform(pws[i], Rcw, tcw, True)
        dpc_dpw_numerical = numerical_derivative(transform, [pws[i], Rcw, tcw], 0)
        print("check dpc%d_dpw%d: " % (i, i), check(dpc_dpw_numerical, dpc_dpws[i]))

        us[i], du_dpcs[i] = project(pcs[i], fx, fy, cx, cy, True)
        du_dpc_numerical = numerical_derivative(project, [pcs[i], fx, fy, cx, cy], 0)
        print("check du%d_dpc%d: " % (i, i), check(du_dpc_numerical, du_dpcs[i]))

        # step2. Calcuate the 3d Gaussian.
        cov3ds[i], dcov3d_drots[i], dcov3d_dscales[i] = compute_cov_3d(gs['rot'][i], gs['scale'][i], True)
        dcov3d_dq_numerical = numerical_derivative(compute_cov_3d, [gs['rot'][i], gs['scale'][i]], 0)
        dcov3d_ds_numerical = numerical_derivative(compute_cov_3d, [gs['rot'][i], gs['scale'][i]], 1)
        print("check dcov3d%d_dq%d: " % (i, i), check(dcov3d_dq_numerical, dcov3d_drots[i]))
        print("check dcov3d%d_ds%d: " % (i, i), check(dcov3d_ds_numerical, dcov3d_dscales[i]))

        cov2ds[i], dcov2d_dcov3ds[i], dcov2d_dpcs[i] = compute_cov_2d(cov3ds[i], pcs[i], Rcw, fx, fy, True)
        dcov2d_dcov3d_numerical = numerical_derivative(compute_cov_2d, [cov3ds[i], pcs[i], Rcw, fx, fy], 0)
        dcov2d_dpc_numerical = numerical_derivative(compute_cov_2d, [cov3ds[i], pcs[i], Rcw, fx, fy], 1)

        print("check dcov2d%d_dcov3d%d: " % (i, i), check(dcov2d_dcov3d_numerical, dcov2d_dcov3ds[i]))
        print("check dcov2d%d_dpc%d: " % (i, i), check(dcov2d_dpc_numerical, dcov2d_dpcs[i]))

        # step3. Project the 3D Gaussian to 2d image as a 2d Gaussian.
        colors[i], dcolor_dshs[i] = sh2color(gs['sh'][i], pws[i], cam_center, True)
        sh2color(gs['sh'][i], pws[i], cam_center)
        sh2color(gs['sh'][i], pws[i]+np.array([1, 0, 0]), cam_center)

        dcolor_dsh_numerical = numerical_derivative(
            sh2color, [gs['sh'][i], pws[i], cam_center], 0)
        dcolor_dpc_numerical = numerical_derivative(
            sh2color, [gs['sh'][i], pws[i], cam_center], 1)
        print("check dcolor%d_dsh%d: " % (i, i), check(
            dcolor_dsh_numerical[0, range(0, 48, 3)], dcolor_dshs[i]))
    # ---------------------------------
    idx = np.argsort(pcs[:, 2])
    idxb = np.argsort(idx)
    colors = colors[idx].reshape(-1)
    cov2ds = cov2ds[idx].reshape(-1)
    alphas = gs['alpha'][idx]
    us = us[idx].reshape(-1)
    x = np.array([16, 8])

    calc_gamma(alphas, cov2ds, colors, us, np.array([16, 8]))

    cov2d0 = cov2ds[:3]
    cinv2d0, dcov2d_dcinv2d = calc_cinv2d(cov2d0, True)
    dcov2d_dcinv2d_numerial = numerical_derivative(calc_cinv2d, [cov2d0], 0)
    print("check dcov2d_dcinv2d: ", check(
        dcov2d_dcinv2d_numerial, dcov2d_dcinv2d))

    cinv2ds = calc_cinv2d(cov2ds)
    alpha0, u0 = alphas[:1], us[:2]
    calc_alpha_prime(alpha0, cinv2d0, u0, x)

    dalphaprime_dalpha_numerial = numerical_derivative(
        calc_alpha_prime, [alpha0, cinv2d0, u0, x], 0)
    dalphaprime_dcinv2d_numerial = numerical_derivative(
        calc_alpha_prime, [alpha0, cinv2d0, u0, x], 1)
    dalphaprime_du_numerial = numerical_derivative(
        calc_alpha_prime, [alpha0, cinv2d0, u0, x], 2)
    alpha_prime, dalphaprime_dalpha, dalphaprime_dcinv2d, dalphaprime_du = calc_alpha_prime(
        alpha0, cinv2d0, u0, x, True)

    print("check dalphaprime_dalpha: ", check(
        dalphaprime_dalpha_numerial, dalphaprime_dalpha))
    print("check dalphaprime_dcinv2d: ", check(
        dalphaprime_dcinv2d_numerial, dalphaprime_dcinv2d))
    print("check dalphaprime_du: ", check(
        dalphaprime_du_numerial, dalphaprime_du))

    gamma, dgamma_dalpha, dgamma_dcov2d, dgamma_dcolor, dgamma_du, _ = calc_gamma(
        alphas, cov2ds, colors, us, x, True)
    dgamma_dalpha_numerial = numerical_derivative(
        calc_gamma, [alphas, cov2ds, colors, us, x], 0)
    dgamma_dcov2d_numerial = numerical_derivative(
        calc_gamma, [alphas, cov2ds, colors, us, x], 1)
    dgamma_dcolor_numerial = numerical_derivative(
        calc_gamma, [alphas, cov2ds, colors, us, x], 2)
    dgamma_du_numerial = numerical_derivative(
        calc_gamma, [alphas, cov2ds, colors, us, x], 3)

    for i in range(gs_num):
        print("check dgamma_dalpha_%d: " % i, check(
            dgamma_dalpha_numerial[:, i], dgamma_dalpha[i].reshape(-1)))
        print("check dgamma_dcov2d_%d: " % i, check(
            dgamma_dcov2d_numerial[:, 3*i:3*i+3], dgamma_dcov2d[i]))
        print("check dgamma_dcolor_%d: " % i, check(
            dgamma_dcolor_numerial[:, 3*i:3*i+3], dgamma_dcolor[i]))
        print("check dgamma_du_%d: " % i, check(
            dgamma_du_numerial[:, 2*i:2*i+2], dgamma_du[i]))

    loss, dloss_dalphas, dloss_dcov2ds, dloss_dcolors, dloss_dus = calc_loss(
        alphas, cov2ds, colors, us, width, height, True)
    dloss_dalpha_numerial = numerical_derivative(
        calc_loss, [alphas, cov2ds, colors, us, width, height], 0)
    dloss_dcov2d_numerial = numerical_derivative(
        calc_loss, [alphas, cov2ds, colors, us, width, height], 1)
    dloss_dcolor_numerial = numerical_derivative(
        calc_loss, [alphas, cov2ds, colors, us, width, height], 2)
    dloss_du_numerial = numerical_derivative(
        calc_loss, [alphas, cov2ds, colors, us, width, height], 3)

    print("check dloss_dalpha: ", check(dloss_dalpha_numerial, dloss_dalphas))
    print("check dloss_dcov2d: ", check(dloss_dcov2d_numerial, dloss_dcov2ds))
    print("check dloss_dcolor: ", check(dloss_dcolor_numerial, dloss_dcolors))
    print("check dloss_du: ", check(dloss_du_numerial, dloss_dus))

    dloss_dcov2ds = dloss_dcov2ds.reshape([gs_num, 1, 3])
    dloss_dalphas = dloss_dalphas.reshape([gs_num, 1, 1])
    dloss_dcolors = dloss_dcolors.reshape([gs_num, 1, 3])
    dloss_dus = dloss_dus.reshape([gs_num, 1, 2])

    dloss_drots = dloss_dcov2ds @ dcov2d_dcov3ds @ dcov3d_drots
    dloss_dscales = dloss_dcov2ds @ dcov2d_dcov3ds @ dcov3d_dscales
    dloss_dshs = dloss_dcolors @ dcov2d_dcov3ds
    dloss_dalphas = dloss_dalphas
    dloss_dpws = dloss_dus @ du_dpcs @ dpc_dpws + \
                 dloss_dcolors @ dcolor_dpc_numerical @ dpc_dpws + \
                 dloss_dcov2ds @ dcov2d_dpcs @ dpc_dpws
                 

    # gs_num = gs.shape[0]
    # print("dloss_du:\n", dloss_du.reshape([gs_num, 2])[idxb])
    # print("dloss_dcov2d:\n", dloss_dcov2d.reshape([gs_num, 3])[idxb])
    # print("dloss_dalpha:\n", dloss_dalpha.reshape([gs_num, 1])[idxb])
    # print("dloss_dcolor:\n", dloss_dcolor.reshape([gs_num, 3])[idxb])
