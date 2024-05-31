import matplotlib.pyplot as plt
import numpy as np
import time
from gsplat.sh_coef import *


def projection_matrix(focal_x, focal_y, width, height, z_near=0.1, z_far=100):
    P = np.zeros([4, 4])
    P[0, 0] = 2 * focal_x / width
    P[1, 1] = 2 * focal_x / width
    P[2, 2] = -(z_near + z_far) / (z_far - z_near)
    P[2, 3] = -(2 * z_near * z_far) / (z_far - z_near)
    P[3, 0] = -1
    return P


def upper_triangular(mat):
    s = mat[0].shape[0]
    n = 0
    if (s == 2):
        n = 3
    elif(s == 3):
        n = 6
    else:
        raise NotImplementedError("no supported mat")
    upper = np.zeros([mat.shape[0], n])
    n = 0
    for i in range(s):
        for j in range(i, s):
            upper[:, n] = mat[:, i, j]
            n = n + 1
    return upper


def symmetric_matrix(upper):
    n = upper.shape[1]
    if (n == 6):
        s = 3
    elif(n == 3):
        s = 2
    else:
        raise NotImplementedError("no supported mat")

    mat = np.zeros([upper.shape[0], s, s])

    n = 0
    for i in range(s):
        for j in range(i, s):
            mat[:, i, j] = upper[:, n]
            if (i != j):
                mat[:, j, i] = upper[:, n]
            n = n + 1
    return mat


def sh2color(sh, pw, twc):
    sh_dim = sh.shape[1]
    color = SH_C0_0 * sh[:, 0:3] + 0.5
    if (sh_dim <= 3):
        return color
    ray_dir = pw - twc
    ray_dir /= np.linalg.norm(ray_dir, axis=1)[:, np.newaxis]
    x = ray_dir[:, 0][:, np.newaxis]
    y = ray_dir[:, 1][:, np.newaxis]
    z = ray_dir[:, 2][:, np.newaxis]

    color = color + \
        SH_C1_0 * y * sh[:, 3:6] + \
        SH_C1_1 * z * sh[:, 6:9] + \
        SH_C1_2 * x * sh[:, 9:12]

    if (sh_dim <= 12):
        return color
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    yz = y * z
    xz = x * z
    color = color + \
        SH_C2_0 * xy * sh[:, 12:15] + \
        SH_C2_1 * yz * sh[:, 15:18] + \
        SH_C2_2 * (2.0 * zz - xx - yy) * sh[:, 18:21] + \
        SH_C2_3 * xz * sh[:, 21:24] + \
        SH_C2_4 * (xx - yy) * sh[:, 24:27]

    if (sh_dim <= 27):
        return color

    color = color +  \
        SH_C3_0 * y * (3.0 * xx - yy) * sh[:, 27:30] + \
        SH_C3_1 * xy * z * sh[:, 30:33] + \
        SH_C3_2 * y * (4.0 * zz - xx - yy) * sh[:, 33:36] + \
        SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[:, 36:39] + \
        SH_C3_4 * x * (4.0 * zz - xx - yy) * sh[:, 39:42] + \
        SH_C3_5 * z * (xx - yy) * sh[:, 42:45] + \
        SH_C3_6 * x * (xx - 3.0 * yy) * sh[:, 45:48]

    return color


def compute_cov_3d(scale, rot):
    # Create scaling matrix
    S = np.zeros([scale.shape[0], 3, 3])
    S[:, 0, 0] = scale[:, 0]
    S[:, 1, 1] = scale[:, 1]
    S[:, 2, 2] = scale[:, 2]

    # Normalize quaternion to get valid rotation
    q = rot
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    # Compute rotation matrix from quaternion
    R = np.array([
        [1.0 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x * z + y * w)],
        [2*(x*y + z*w), 1.0 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1.0 - 2*(x**2 + y**2)]
    ]).transpose(2, 0, 1)
    M = R @ S

    # Compute 3D world covariance matrix Sigma
    Sigma = M @ M.transpose(0, 2, 1)
    cov3d = upper_triangular(Sigma)

    return cov3d


def compute_cov_2d(pc, fx, fy, width, height, cov3d, Rcw):
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]

    tan_fovx = 2 * np.arctan(width/(2*fx))
    tan_fovy = 2 * np.arctan(height/(2*fy))

    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    x = np.clip(x / z, -limx, limx) * z
    y = np.clip(y / z, -limy, limy) * z

    J = np.zeros([pc.shape[0], 3, 3])
    z2 = z * z
    J[:, 0, 0] = fx / z
    J[:, 0, 2] = -(fx * x) / z2
    J[:, 1, 1] = fy / z
    J[:, 1, 2] = -(fy * y) / z2

    T = J @ Rcw

    Sigma = symmetric_matrix(cov3d)

    Sigma_prime = T @ Sigma @ T.transpose(0, 2, 1)
    Sigma_prime[:, 0, 0] += 0.3
    Sigma_prime[:, 1, 1] += 0.3

    cov2d = upper_triangular(Sigma_prime[:, :2, :2])
    # cov2d[out_idx] = 0
    return cov2d


def project(pw, Rcw, tcw, fx, fy, cx, cy):
    # project the mean of 2d gaussian to image.
    # forward.md (1.1) (1.2)
    pc = (Rcw @ pw.T).T + tcw
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]
    u = np.stack([(x * fx / z + cx),
                  (y * fy / z + cy)], axis=1)
    return u, pc


def inverse_cov2d(cov2d):
    # forward.md 5.3
    # compute inverse of cov2d
    det_inv = 1. / (cov2d[:, 0] * cov2d[:, 2] - cov2d[:, 1] * cov2d[:, 1] + 0.000001)
    cinv2d = np.array([cov2d[:, 2] * det_inv, -cov2d[:, 1] * det_inv, cov2d[:, 0] * det_inv]).T
    areas = 3 * np.sqrt(np.vstack([cov2d[:, 0], cov2d[:, 2]])).T
    return cinv2d, areas.astype(np.int32)


def splat(height, width, us, cinv2d, alpha, depth, color, areas, im=None):
    image = np.zeros([height, width, 3])
    image_T = np.ones([height, width])

    start = time.time()

    # sort by depth
    sort_idx = np.argsort(depth)

    idx_map = np.array((np.meshgrid(np.arange(0, width), np.arange(0, height))))
    win_size = np.array([width, height])

    for j, i in enumerate(sort_idx):
        if (j % 10000 == 0):
            print("processing... %3.f%%" % (j / float(us.shape[0]) * 100.))
            if (im is not None):
                im.set_data(image)
                plt.pause(0.1)

        if (depth[i] < 0.2 or depth[i] > 100):
            continue

        u = us[i]
        if (np.any(np.abs(u / win_size) > 1.3)):
            continue

        r = areas[i]
        x0 = int(np.maximum(np.minimum(u[0] - r[0], width), 0))
        x1 = int(np.maximum(np.minimum(u[0] + r[0], width), 0))
        y0 = int(np.maximum(np.minimum(u[1] - r[1], height), 0))
        y1 = int(np.maximum(np.minimum(u[1] + r[1], height), 0))

        if ((x1 - x0) * (y1 - y0) == 0):
            continue

        cinv = cinv2d[i]
        opa = alpha[i]
        patch_color = color[i]

        d = u[:, np.newaxis, np.newaxis] - idx_map[:, y0:y1, x0:x1]
        # mahalanobis distance
        maha_dist = cinv[0] * d[0] * d[0] + cinv[2] * d[1] * d[1] + 2 * cinv[1] * d[0] * d[1]
        patch_alpha = np.exp(-0.5 * maha_dist) * opa
        patch_alpha[patch_alpha > 0.99] = 0.99

        # draw inverse gaussian
        # th = 0.7
        # patch_alpha = np.exp(-0.5 * maha_dist) * opa
        # patch_alpha[patch_alpha <= th] = 0
        # patch_alpha[patch_alpha > th] = (1 - patch_alpha[patch_alpha > th])

        T = image_T[y0:y1, x0:x1]
        image[y0:y1, x0:x1, :] += (patch_alpha * T)[:, :, np.newaxis] * patch_color
        image_T[y0:y1, x0:x1] = T * (1 - patch_alpha)
    end = time.time()
    time_diff = end - start
    print("add patch time %f\n" % time_diff)
    if (im is not None):
        im.set_data(image)
        plt.pause(0.1)
    return image
