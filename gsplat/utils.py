import torch
import gsplatcu as gsc
import collections
import numpy as np


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    this file is copied from Plenoxels
    https://github.com/sxyu/svox2/blob/ee80e2c4df8f29a407fda5729a494be94ccf9234/opt/util/util.py#L78

    Continuous learning rate decay function. Adapted from JaxNeRF

    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.

    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


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


def rainbow(scalars, scalar_min=0, scalar_max=255):
    range = scalar_max - scalar_min
    values = 1.0 - (scalars - scalar_min) / range
    # values = (scalars - scalar_min) / range  # using inverted color
    colors = torch.zeros([scalars.shape[0], 3], dtype=torch.float32, device='cuda')
    values = torch.clip(values, 0, 1)

    h = values * 5.0 + 1.0
    i = torch.floor(h).to(torch.int32)
    f = h - i
    f[torch.logical_not(i % 2)] = 1 - f[torch.logical_not(i % 2)]
    n = 1 - f

    # idx = i <= 1
    colors[i <= 1, 0] = n[i <= 1]
    colors[i <= 1, 1] = 0
    colors[i <= 1, 2] = 1

    colors[i == 2, 0] = 0
    colors[i == 2, 1] = n[i == 2]
    colors[i == 2, 2] = 1

    colors[i == 3, 0] = 0
    colors[i == 3, 1] = 1
    colors[i == 3, 2] = n[i == 3]

    colors[i == 4, 0] = n[i == 4]
    colors[i == 4, 1] = 1
    colors[i == 4, 2] = 0

    colors[i >= 5, 0] = 1
    colors[i >= 5, 1] = n[i >= 5]
    colors[i >= 5, 2] = 0
    shs = (colors - 0.5) / 0.28209479177387814
    return shs


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


def get_shs(low_shs, high_shs):
    return torch.cat((low_shs, high_shs), dim=1)