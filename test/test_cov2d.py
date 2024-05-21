import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import sys
import os


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


def calc_cov2d(cov3d, pc, Rcw, fx, fy, calc_J=False):
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
        dm_dpc =\
            np.array([[-fx*r20/z**2, 0, -fx*r00/z**2 + 2*fx*r20*x/z**3],
                      [-fx*r21/z**2, 0, -fx*r01/z**2 + 2*fx*r21*x/z**3],
                      [-fx*r22/z**2, 0, -fx*r02/z**2 + 2*fx*r22*x/z**3],
                      [0, -fy*r20/z**2, -fy*r10/z**2 + 2*fy*r20*y/z**3],
                      [0, -fy*r21/z**2, -fy*r11/z**2 + 2*fy*r21*y/z**3],
                      [0, -fy*r22/z**2, -fy*r12/z**2 + 2*fy*r22*y/z**3]])
        return cov2d, dcov2d_dcov3d, dcov2d_dm @ dm_dpc
    else:
        return cov2d


def get_J(pc, fx, fy):
    x, y, z = pc
    z2 = z*z
    return np.array([[fx/z, 0, -fx*x/z2],
                     [0, fy/z, -fy*y/z2]])


if __name__ == "__main__":
    cov3d = np.array([1.24892526, -2.73532296,  0.86639549,
                     7.97665233, -3.00404921, 2.70966732])
    Rcw = np.array([[-0.267058, -0.302404, -0.916068],
                    [0.308444,  0.872984, -0.378096],
                    [0.914052, -0.382944, -0.140058]])
    pc = np.array([1, 2, 3.])
    fx, fy = 200, 100

    cov2d, dcov2d_dcov3d, dcov2d_dpc = calc_cov2d(cov3d, pc, Rcw, fx, fy, True)
    dcov2d_dcov3d_numerical = numerical_derivative(calc_cov2d, [cov3d, pc, Rcw, fx, fy], 0)
    dcov2d_dpc_numerical = numerical_derivative(calc_cov2d, [cov3d, pc, Rcw, fx, fy], 1)

    print(np.max(np.abs(dcov2d_dcov3d_numerical - dcov2d_dcov3d)))
    print(np.max(np.abs(dcov2d_dpc_numerical - dcov2d_dpc)))

