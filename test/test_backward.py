import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from gaussian_splatting import *


def calc_tau(alpha_prime):
    tau = [1.]
    for a in alpha_prime:
        tau.append(tau[-1] * (1-a))
    return np.array(tau)


def calc_gamma(alphas, cov2ds, colors, us, x, calc_J=False):
    cov2ds = cov2ds.reshape([-1, 3])
    colors = colors.reshape([-1, 3])
    us = us.reshape([-1, 2])
    tau = 1.
    gamma = np.zeros(3)
    for alpha, cov2d, color, u in zip(alphas, cov2ds, colors, us):
        cinv2d = calc_cinv2d(cov2d)
        alpha_prime = calc_alpha_prime(alpha, cinv2d, u, x)
        gamma += alpha_prime * color * tau
        tau = tau * (1 - alpha_prime)
    if (calc_J):
        gamma_cur2last = np.zeros(3)
        reversed_values = reversed(list(zip(alphas, cov2ds, colors, us)))
        dgamma_dalpha = []
        dgamma_dcov2d = []
        dgamma_dcolor = []
        dgamma_du = []
        for alpha, cov2d, color, u in reversed_values:
            cinv2d, dcinv2d_dcov2d = calc_cinv2d(cov2d, True)
            alpha_prime, dalphaprime_dalpha, dalphaprime_dcinv2d, dalphaprime_du =\
                calc_alpha_prime(alpha, cinv2d, u, x, True)
            tau = tau / (1 - alpha_prime)
            dgamma_dalphaprime = (tau * (color - gamma_cur2last)).reshape([3, 1])
            dgamma_dalpha.insert(0, dgamma_dalphaprime @ dalphaprime_dalpha)
            dgamma_dcov2d.insert(0, dgamma_dalphaprime @ dalphaprime_dcinv2d @ dcinv2d_dcov2d)
            dgamma_dcolor.insert(0, tau * alpha_prime * np.eye(3))
            dgamma_du.insert(0, dgamma_dalphaprime @ dalphaprime_du)
            gamma_cur2last = alpha_prime * color + (1 - alpha_prime) * gamma_cur2last
        return gamma, dgamma_dalpha, dgamma_dcov2d, dgamma_dcolor, dgamma_du
    else:
        return gamma


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
    maha_dist = cinv2d[0] * d[0] * d[0] + cinv2d[2] * d[1] * d[1] + 2 * cinv2d[1] * d[0] * d[1]
    g = np.exp(-0.5 * maha_dist)
    alphaprime = g * alpha
    if (calc_J):
        dalphaprime_dalpha = np.array([[g]])
        dalphaprime_dcinv2d = -0.5 * alphaprime * np.array([d[0] * d[0], 2 * d[0] * d[1], d[1] * d[1]])
        dalphaprime_du = alphaprime * np.array([-cinv2d[0]*d[0] - cinv2d[1]*d[1], (-cinv2d[1]*d[0] - cinv2d[2]*d[1])])
        return alphaprime, dalphaprime_dalpha, dalphaprime_dcinv2d.reshape([1, 3]), dalphaprime_du.reshape([1, 2])
    else:
        return alphaprime


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
                        ], dtype=np.float64)

    dtypes = [('pos', '<f8', (3,)),
              ('rot', '<f8', (4,)),
              ('scale', '<f8', (3,)),
              ('alpha', '<f8'),
              ('sh', '<f8', (3,))]

    gs = np.frombuffer(gs_data.tobytes(), dtype=dtypes)

    # Camera info
    tcw = np.array([1.03796196, 0.42017467, 4.67804612])
    Rcw = np.array([[0.89699204,  0.06525223,  0.43720409],
                    [-0.04508268,  0.99739184, -0.05636552],
                    [-0.43974177,  0.03084909,  0.89759429]]).T

    W = int(32)  # 1957  # 979
    H = int(32)  # 1091  # 546
    focal_x = 16
    focal_y = 16

    K = np.array([[focal_x, 0, W/2.],
                  [0, focal_y, H/2.],
                  [0, 0, 1.]])

    Tcw = np.eye(4)
    Tcw[:3, :3] = Rcw
    Tcw[:3, 3] = tcw
    cam_center = np.linalg.inv(Tcw)[:3, 3]

    pw = gs['pos']

    # step1. Transform pw to camera frame,
    # and project it to iamge.
    u, pc = project(pw, Tcw, K)

    depth = pc[:, 2]

    # step2. Calcuate the 3d Gaussian.
    cov3d = compute_cov_3d(gs['scale'], gs['rot'])

    # step3. Project the 3D Gaussian to 2d image as a 2d Gaussian.
    cov2d = compute_cov_2d(pc, focal_x, focal_y, cov3d, Rcw)

    # step4. get color info
    ray_dir = pw[:, :3] - cam_center
    ray_dir /= np.linalg.norm(ray_dir, axis=1)[:, np.newaxis]
    color = sh2color(gs['sh'], ray_dir)

    # ---------------------------------
    idx = np.argsort(pc[:, 2])
    color = color[idx].reshape(-1)
    cov2d = cov2d[idx].reshape(-1)
    alpha = gs['alpha'][idx]
    u = u[idx].reshape(-1)
    x = np.array([16, 16])

    cov2d0 = cov2d[:3]
    cinv2d0, dcov2d_dcinv2d = calc_cinv2d(cov2d0, True)
    dcov2d_dcinv2d_numerial = numerical_derivative(calc_cinv2d, [cov2d0], 0)
    print("check dcov2d_dcinv2d: ", check(dcov2d_dcinv2d_numerial, dcov2d_dcinv2d))

    cinv2d = calc_cinv2d(cov2d)
    alpha0, u0 = alpha[:1], u[:2]
    calc_alpha_prime(alpha0, cinv2d0, u0, x)

    dalphaprime_dalpha_numerial = numerical_derivative(calc_alpha_prime, [alpha0, cinv2d0, u0, x], 0)
    dalphaprime_dcinv2d_numerial = numerical_derivative(calc_alpha_prime, [alpha0, cinv2d0, u0, x], 1)
    dalphaprime_du_numerial = numerical_derivative(calc_alpha_prime, [alpha0, cinv2d0, u0, x], 2)
    alpha_prime, dalphaprime_dalpha, dalphaprime_dcinv2d, dalphaprime_du = calc_alpha_prime(alpha0, cinv2d0, u0, x, True)

    print("check dalphaprime_dalpha: ", check(dalphaprime_dalpha_numerial, dalphaprime_dalpha))
    print("check dalphaprime_dcinv2d: ", check(dalphaprime_dcinv2d_numerial, dalphaprime_dcinv2d))
    print("check dalphaprime_du: ", check(dalphaprime_du_numerial, dalphaprime_du))

    gamma, dgamma_dalpha, dgamma_dcov2d, dgamma_dcolor, dgamma_du = calc_gamma(alpha, cov2d, color, u, x, True)
    dgamma_dalpha_numerial = numerical_derivative(calc_gamma, [alpha, cov2d, color, u, x], 0)
    dgamma_dcov2d_numerial = numerical_derivative(calc_gamma, [alpha, cov2d, color, u, x], 1)
    dgamma_dcolor_numerial = numerical_derivative(calc_gamma, [alpha, cov2d, color, u, x], 2)
    dgamma_du_numerial = numerical_derivative(calc_gamma, [alpha, cov2d, color, u, x], 3)

    for i in range(alpha.shape[0]):
        print("check dgamma_dalpha_%d: " % i, check(dgamma_dalpha_numerial[:, i], dgamma_dalpha[i].reshape(-1)))
        print("check dgamma_dcov2d_%d: " % i, check(dgamma_dcov2d_numerial[:, 3*i:3*i+3], dgamma_dcov2d[i]))
        print("check dgamma_dcolor_%d: " % i, check(dgamma_dcolor_numerial[:, 3*i:3*i+3], dgamma_dcolor[i]))
        print("check dgamma_du_%d: " % i, check(dgamma_du_numerial[:, 2*i:2*i+2], dgamma_du[i]))

    for i in range(alpha.shape[0]):
        # print("dgamma_dalpha_%d\n: " % i, dgamma_dalpha[i])
        print("dgamma_du_%d\n: " % i, dgamma_du[i])
