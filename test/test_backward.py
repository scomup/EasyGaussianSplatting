import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from sh_coef import *
from backward_cpu import *


if __name__ == "__main__":
    gs_data = np.random.rand(4, 59)
    gs_data0 = np.array([[0.,  0.,  0.,  # xyz
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

    gs_data[:, :14] = gs_data0
    dtypes = [('pos', '<f8', (3,)),
              ('rot', '<f8', (4,)),
              ('scale', '<f8', (3,)),
              ('alpha', '<f8'),
              ('sh', '<f8', (48,))]
    gs = np.frombuffer(gs_data.tobytes(), dtype=dtypes)
    gs_num = gs.shape[0]

    # Camera info
    tcw = np.array([1.03796196, 0.42017467, 4.67804612])
    Rcw = np.array([[0.89699204,  0.06525223,  0.43720409],
                    [-0.04508268,  0.99739184, -0.05636552],
                    [-0.43974177,  0.03084909,  0.89759429]]).T
    twc = np.linalg.inv(Rcw) @ (-tcw)
    width = int(32)
    height = int(16)
    fx = 16
    fy = 16
    cx = width/2.
    cy = height/2.
    image_gt = np.zeros([height, width, 3])

    loss, dloss_drots, dloss_dscales, dloss_dshs, dloss_dalphas, dloss_dpws = backward(
        gs['rot'], gs['scale'], gs['sh'], gs['alpha'], gs['pos'], Rcw, tcw, fx, fy, cx, cy, image_gt, True)
    print(dloss_dscales)
