import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import sys
import os


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
    S = np.diag(s)
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


def calc_cov(q, s, calc_J=False):
    m, dm_mq, dm_ds = calc_m(q, s, True)
    cov, dcov_dm = calc_mmt(m, True)
    if (calc_J):
        return cov, dcov_dm @ dm_mq, dcov_dm @ dm_ds
    else:
        return cov


def calc_rot(q, calc_J=False):
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    R = np.array([
        [1.0 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x * z + y * w)],
        [2*(x*y + z*w), 1.0 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1.0 - 2*(x**2 + y**2)]
    ]).reshape(-1)
    if (calc_J):
        drot_dq = np.array([[0, 0, -4*y, -4*z],
                            [-2*z, 2*y, 2*x, -2*w],
                            [2*y, 2*z, 2*w, 2*x],
                            [2*z, 2*y, 2*x, 2*w],
                            [0, -4*x, 0, -4*z],
                            [-2*x, -2*w, 2*z, 2*y],
                            [-2*y, 2*z, -2*w, 2*x],
                            [2*x, 2*w, 2*z, 2*y],
                            [0, -4*x, -4*y, 0]])
        return R, drot_dq
    else:
        return R

if __name__ == "__main__":
    q = np.array([0.606, -0.002, -0.755, 0.252])
    s = np.array([1.2, 3.2, 0.5])

    dcov_dq_numerical = numerical_derivative(calc_cov, [q, s], 0)
    dcov_ds_numerical = numerical_derivative(calc_cov, [q, s], 1)
    cov, dcov_dq, dcov_ds = calc_cov(q, s, True)

    print(np.max(np.abs(dcov_dq_numerical - dcov_dq)))
    print(np.max(np.abs(dcov_ds - dcov_ds)))
