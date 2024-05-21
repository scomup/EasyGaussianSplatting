from sympy import symbols, Matrix, diff

m00, m01, m02, m10, m11, m12 = symbols("m00, m01, m02, m10, m11, m12")
r00, r01, r02, r10, r11, r12, r20, r21, r22 = symbols("r00, r01, r02, r10, r11, r12, r20, r21, r22")

R = Matrix([[r00, r01, r02],
            [r10, r11, r12],
            [r20, r21, r22]])


a, b, c, d, e, f = symbols("a, b, c, d, e, f")
f_x, f_y = symbols("f_x, f_y")
x, y, z = symbols("x, y, z")

z2 = z*z
J = Matrix([[f_x/z, 0, -f_x*x/z2],
            [0, f_y/z, -f_y*y/z2]])

JR = J*R

djr_dpc = Matrix([
    [diff(JR[0, 0], x), diff(JR[0, 0], y), diff(JR[0, 0], z)],
    [diff(JR[0, 1], x), diff(JR[0, 1], y), diff(JR[0, 1], z)],
    [diff(JR[0, 2], x), diff(JR[0, 2], y), diff(JR[0, 2], z)],
    [diff(JR[1, 0], x), diff(JR[1, 0], y), diff(JR[1, 0], z)],
    [diff(JR[1, 1], x), diff(JR[1, 1], y), diff(JR[1, 1], z)],
    [diff(JR[1, 2], x), diff(JR[1, 2], y), diff(JR[1, 2], z)],
    ])


S = Matrix([[a, b, c],
            [b, d, e],
            [c, e, f]])

M = Matrix([[m00, m01, m02],
            [m10, m11, m12]])

Cov = M * S * M.T

cov0 = Cov[0, 0]
cov1 = Cov[0, 1]
cov2 = Cov[1, 1]

dcov2d_dcov3d = Matrix([
    [diff(cov0, a), diff(cov0, b), diff(cov0, c), diff(cov0, d), diff(cov0, e), diff(cov0, f)],
    [diff(cov1, a), diff(cov1, b), diff(cov1, c), diff(cov1, d), diff(cov1, e), diff(cov1, f)],
    [diff(cov2, a), diff(cov2, b), diff(cov2, c), diff(cov2, d), diff(cov2, e), diff(cov2, f)]])

dcov2d_dm = Matrix([
    [diff(cov0, m00), diff(cov0, m01), diff(cov0, m02), diff(cov0, m10), diff(cov0, m11), diff(cov0, m12)],
    [diff(cov1, m00), diff(cov1, m01), diff(cov1, m02), diff(cov1, m10), diff(cov1, m11), diff(cov1, m12)],
    [diff(cov2, m00), diff(cov2, m01), diff(cov2, m02), diff(cov2, m10), diff(cov2, m11), diff(cov2, m12)]])


dm_s = Matrix.zeros(9, 3)
dm_s[0, 0] = 1.0 - 2*(y**2 + z**2)
dm_s[1, 1] = 2*(x*y - z*w)
dm_s[2, 2] = 2*(x * z + y * w)
dm_s[3, 0] = 2*(x*y + z*w)
dm_s[4, 1] = 1.0 - 2*(x**2 + z**2),
dm_s[5, 2] = 2*(y*z - x*w)
dm_s[6, 0] = 2*(x*z - y*w)
dm_s[7, 1] = 2*(y*z + x*w)
dm_s[8, 2] = 1.0 - 2*(x**2 + y**2)

a, b, c, d, e, f, g, h, i = symbols("a,b,c,d,e,f,g,h,i")
M = Matrix([[a, b, c],
            [d, e, f],
            [g, h, i]])
MMT = M*M.T



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
    q = np.array([0.606, -0.002, -0.755, 0.252])
    s = np.array([1.2, 3.2, 0.5])
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
