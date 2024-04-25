import numpy as np
from plyfile import PlyData


def matrix_to_quaternion(matrices):
    m00, m01, m02 = matrices[:, 0, 0], matrices[:, 0, 1], matrices[:, 0, 2]
    m10, m11, m12 = matrices[:, 1, 0], matrices[:, 1, 1], matrices[:, 1, 2]
    m20, m21, m22 = matrices[:, 2, 0], matrices[:, 2, 1], matrices[:, 2, 2]
    t = 1 + m00 + m11 + m22
    s = np.ones_like(m00)
    w = np.ones_like(m00)
    x = np.ones_like(m00)
    y = np.ones_like(m00)
    z = np.ones_like(m00)

    t_positive = t > 0.0000001
    s[t_positive] = 0.5 / np.sqrt(t[t_positive])
    w[t_positive] = 0.25 / s[t_positive]
    x[t_positive] = (m21[t_positive] - m12[t_positive]) * s[t_positive]
    y[t_positive] = (m02[t_positive] - m20[t_positive]) * s[t_positive]
    z[t_positive] = (m10[t_positive] - m01[t_positive]) * s[t_positive]

    c1 = np.logical_and(m00 > m11, m00 > m22)
    cond1 = np.logical_and(np.logical_not(t_positive), np.logical_and(m00 > m11, m00 > m22))

    s[cond1] = 2.0 * np.sqrt(1.0 + m00[cond1] - m11[cond1] - m22[cond1])
    w[cond1] = (m21[cond1] - m12[cond1]) / s[cond1]
    x[cond1] = 0.25 * s[cond1]
    y[cond1] = (m01[cond1] + m10[cond1]) / s[cond1]
    z[cond1] = (m02[cond1] + m20[cond1]) / s[cond1]

    c2 = np.logical_and(np.logical_not(c1), m11 > m22)
    cond2 = np.logical_and(np.logical_not(t_positive), c2)
    s[cond2] = 2.0 * np.sqrt(1.0 + m11[cond2] - m00[cond2] - m22[cond2])
    w[cond2] = (m02[cond2] - m20[cond2]) / s[cond2]
    x[cond2] = (m01[cond2] + m10[cond2]) / s[cond2]
    y[cond2] = 0.25 * s[cond2]
    z[cond2] = (m12[cond2] + m21[cond2]) / s[cond2]

    c3 = np.logical_and(np.logical_not(c1), np.logical_not(c2))
    cond3 = np.logical_and(np.logical_not(t_positive), c3)
    s[cond3] = 2.0 * np.sqrt(1.0 + m22[cond3] - m00[cond3] - m11[cond3])
    w[cond3] = (m10[cond3] - m01[cond3]) / s[cond3]
    x[cond3] = (m02[cond3] + m20[cond3]) / s[cond3]
    y[cond3] = (m12[cond3] + m21[cond3]) / s[cond3]
    z[cond3] = 0.25 * s[cond3]
    return np.array([w, x, y, z]).T


def load_ply(path, T=None):
    max_sh_degree = 3
    plydata = PlyData.read(path)
    pos = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)

    alphas = np.asarray(plydata.elements[0]["opacity"])
    alphas = 1/(1 + np.exp(-alphas))

    scales = np.stack((np.asarray(plydata.elements[0]["scale_0"]),
                       np.asarray(plydata.elements[0]["scale_1"]),
                       np.asarray(plydata.elements[0]["scale_2"])),  axis=1)

    rots = np.stack((np.asarray(plydata.elements[0]["rot_0"]),
                     np.asarray(plydata.elements[0]["rot_1"]),
                     np.asarray(plydata.elements[0]["rot_2"]),
                     np.asarray(plydata.elements[0]["rot_3"])),  axis=1)

    rots /= np.linalg.norm(rots, axis=1)[:, np.newaxis]

    shs = np.zeros([pos.shape[0], 48])
    shs[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    shs[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
    shs[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

    sh_rest_dim = len(plydata.elements[0][0])-17

    for i in range(sh_rest_dim):
        name = "f_rest_%d" % i
        shs[:, 3 + i] = np.asarray(plydata.elements[0][name])

    shs[:, 3:] = shs[:, 3:].reshape(-1, 3, 15).transpose([0, 2, 1]).reshape(-1, 45)

    pos = pos.astype(np.float32)
    rots = rots.astype(np.float32)
    scales = np.exp(scales)
    scales = scales.astype(np.float32)
    alphas = alphas.astype(np.float32)
    shs = shs.astype(np.float32)

    dtypes = [('pos', '<f4', (3,)),
              ('rot', '<f4', (4,)),
              ('scale', '<f4', (3,)),
              ('alpha', '<f4'),
              ('sh', '<f4', (48,))]

    if (T is not None):
        # Transform to world
        pos = (T @ pos.T).T
        w = rots[:, 0]
        x = rots[:, 1]
        y = rots[:, 2]
        z = rots[:, 3]
        R = np.array([
            [1.0 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x * z + y * w)],
            [2*(x*y + z*w), 1.0 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1.0 - 2*(x**2 + y**2)]
        ]).transpose(2, 0, 1)
        R_new = T @ R
        rots = matrix_to_quaternion(R_new)

    gs = np.rec.fromarrays(
        [pos, rots, scales, alphas, shs], dtype=dtypes)

    return gs


if __name__ == "__main__":
    gs = load_ply("/home/liu/workspace/gaussian-splatting/output/fb15ba66-e/point_cloud/iteration_7000/point_cloud.ply")
    print(gs.shape)
