import torch
import gsplatcu as gsc
import collections
import numpy as np


def rotate_vector_by_quaternion(q, v):
    q = torch.nn.functional.normalize(q)
    u = q[:, 1:, np.newaxis]
    s = q[:, 0, np.newaxis, np.newaxis]
    v = v[:, :,  np.newaxis]
    v_prime = 2.0 * u * (u.permute(0, 2, 1) @ v) +\
        v * (s*s - (u.permute(0, 2, 1) @ u)) +\
        2.0 * torch.linalg.cross(u, v, dim=1) * s
    return v_prime.squeeze()


def compute_cov_3d_torch(scale, q):
    # Create scaling matrix
    S = torch.zeros([scale.shape[0], 3, 3], device='cuda')
    S[:, 0, 0] = scale[:, 0]
    S[:, 1, 1] = scale[:, 1]
    S[:, 2, 2] = scale[:, 2]
    # Normalize quaternion to get valid rotation
    q = torch.nn.functional.normalize(q)
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    # Compute rotation matrix from quaternion
    R = torch.stack([
        1.0 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x * z + y * w),
        2*(x*y + z*w), 1.0 - 2*(x**2 + z**2), 2*(y*z - x*w),
        2*(x*z - y*w), 2*(y*z + x*w), 1.0 - 2*(x**2 + y**2)
    ], dim=1).reshape(-1, 3, 3)
    M = R @ S

    # Compute 3D world covariance matrix Sigma
    Sigma = M @ M.permute(0, 2, 1)

    return Sigma


def get_alphas_raw(x):
    """
    inverse of sigmoid
    """
    if isinstance(x, float):
        return np.log(x/(1-x))
    else:
        return torch.log(x/(1-x))


def get_alphas(x):
    return torch.sigmoid(x)


def get_scales_raw(x):
    if isinstance(x, float):
        return np.log(x)
    else:
        return torch.log(x)


def get_scales(x):
    return torch.exp(x)


def get_rots(x):
    return torch.nn.functional.normalize(x)
