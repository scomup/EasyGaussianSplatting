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


def calc_m(rot, s, calc_J=False):
    R = rot.reshape([3, 3])
    S = np.diag(s)
    M = R @ S
    m = M.reshape(-1)
    if (calc_J):
        dm_rot = np.zeros([9, 9])
        dm_rot[0:3, 0:3] = S
        dm_rot[3:6, 3:6] = S
        dm_rot[6:9, 6:9] = S
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
        return m, dmmt_m
    else:
        return mmt


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
        drot_dq = np.zeros([9, 4])
        drot_dq[0, 2] = -4*y
        drot_dq[0, 3] = -4*z
        drot_dq[1, 0] = -2*z
        drot_dq[1, 1] = 2*y
        drot_dq[1, 2] = 2*x
        drot_dq[1, 3] = -2*w
        drot_dq[2, 0] = 2*y
        drot_dq[2, 1] = 2*z
        drot_dq[2, 2] = 2*w
        drot_dq[2, 3] = 2*x
        drot_dq[3, 0] = 2*z
        drot_dq[3, 1] = 2*y
        drot_dq[3, 2] = 2*x
        drot_dq[3, 3] = 2*w
        drot_dq[4, 1] = -4*x
        drot_dq[4, 3] = -4*z
        drot_dq[5, 0] = -2*x
        drot_dq[5, 1] = -2*w
        drot_dq[5, 2] = 2*z
        drot_dq[5, 3] = 2*y
        drot_dq[6, 0] = -2*y
        drot_dq[6, 1] = 2*z
        drot_dq[6, 2] = -2*w
        drot_dq[6, 3] = 2*x
        drot_dq[7, 0] = 2*x
        drot_dq[7, 1] = 2*w
        drot_dq[7, 2] = 2*z
        drot_dq[7, 3] = 2*y
        drot_dq[8, 1] = -4*x
        drot_dq[8, 2] = -4*y
        return R, drot_dq
    else:
        return R

if __name__ == "__main__":
    q = np.array([ 0.606, -0.002, -0.755, 0.252])
    s  = np.array([ 1.2, 3.2, 0.5])
    drot_dq_numerical = numerical_derivative(calc_rot, [q], 0)
    rot, drot_dq = calc_rot(q, True)
    # print(drot_dq_numerical)
    # print(drot_dq)

    dm_drot_numerical = numerical_derivative(calc_m, [rot, s], 0)
    dm_ds_numerical = numerical_derivative(calc_m, [rot, s], 1)
    m, d_mrot, dm_ds = calc_m(rot, s, True)

    dmmt_dm_numerical = numerical_derivative(calc_mmt, [m], 0)
    mmt, dmmt_m = calc_mmt(m, True)
    pass
