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


def expSO2(v):
    return np.array([[np.cos(v), -np.sin(v)],
                     [np.sin(v), np.cos(v)]])


def calc_cov2_sqrt(v, calc_J=False):
    R = expSO2(v[2])
    S = np.diag(v[:2])
    Sigma_sqrt = R @ S
    cov2d_sqrt = np.array([Sigma_sqrt[0, 0], Sigma_sqrt[0, 1], Sigma_sqrt[1, 1]])
    if (calc_J):
        sth = R[0, 0]
        cth = R[1, 0]
        s0 = v[0]
        s1 = v[1]
        dcov2sqrt_dv = np.zeros([3, 3])
        dcov2sqrt_dv[0, 0] = sth
        dcov2sqrt_dv[0, 2] = -s0 * cth
        dcov2sqrt_dv[1, 1] = -cth
        dcov2sqrt_dv[1, 2] = -sth * s1
        dcov2sqrt_dv[2, 1] = sth
        dcov2sqrt_dv[2, 2] = -cth * s1
        return cov2d_sqrt, dcov2sqrt_dv
    else:
        return cov2d_sqrt


def calc_mmt(m, calc_J=False):
    a, b, c = m
    M = np.array([a, b, -b, c]).reshape([2, 2])
    MMT = M @ M.T
    mmt = np.array([MMT[0, 0], MMT[0, 1], MMT[1, 1]])
    if (calc_J):
        a, b, c = cov2_sqrt
        dm_dmmt = np.zeros([3, 3])
        dm_dmmt[0, 0] = 2*a
        dm_dmmt[0, 1] = 2*b
        dm_dmmt[1, 0] = -b
        dm_dmmt[1, 1] = -a + c
        dm_dmmt[1, 2] = b
        dm_dmmt[2, 1] = 2*b
        dm_dmmt[2, 2] = 2*c
        return mmt, dm_dmmt
    else:
        return mmt


def calc_cov2d(v, calc_J=False):
    cov2d_sqrt = calc_cov2_sqrt(v)
    cov2d = calc_mmt(cov2d_sqrt)
    if (calc_J):
        _, dcov2sqrt_dv = calc_cov2_sqrt(v, True)
        _, dcov2d_dcov2sqrt = calc_mmt(cov2d_sqrt, True)
        return cov2d, dcov2d_dcov2sqrt @ dcov2sqrt_dv
    else:
        return cov2d

if __name__ == "__main__":
    v = np.array([-0.1, -2, 0.4])
    cov2_sqrt = calc_cov2_sqrt(v)
    # print(cov2_sqrt)
    dcov2sqrt_dv_numerical = numerical_derivative(calc_cov2_sqrt, [v], 0)
    # print(dcov2sqrt_dv_numerical)
    cov2d_sqrt, dcov2sqrt_dv = calc_cov2_sqrt(v, True)
    # print(dcov2sqrt_dv)
    dcov2d_dcov2sqrt_numerical = numerical_derivative(calc_mmt, [cov2d_sqrt], 0)
    # print(dcov2d_dcov2sqrt_numerical)
    cov2d, dcov2d_dcov2sqrt = calc_mmt(cov2d_sqrt, True)
    # print(dcov2d_dcov2sqrt)

    dcov2d_dv_numerical = numerical_derivative(calc_cov2d, [v], 0)
    cov2d, dcov2d_v = calc_cov2d(v, True)
    print(dcov2d_dv_numerical)
    print(dcov2d_v)

    pass
