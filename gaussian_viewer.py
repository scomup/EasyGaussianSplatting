#!/usr/bin/env python3

import numpy as np
from read_ply import *

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'viewer'))

from viewer import *
from custom_items import *


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", help="the ply path")
    args = parser.parse_args()
    if args.ply:
        # "/home/liu/workspace/gaussian-splatting/output/fb15ba66-e/point_cloud/iteration_30000/point_cloud.ply
        ply_fn = args.ply
        print("Try to load %s ..." % ply_fn)
        cam_2_world = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        gs = load_ply(ply_fn, cam_2_world)
        gs_data = gs.view(np.float32).reshape(gs.shape[0], -1)
    else:
        gs_data = np.array([[0.,  0.,  0.,  # xyz
                            1.,  0.,  0., 0.,  # rot
                            0.05,  0.05,  0.05,  # size
                            1.,
                            1.772484,  -1.772484,  1.772484],
                            [1.,  0.,  0.,
                            1.,  0.,  0., 0.,
                            0.2,  0.05,  0.05,
                            1.,
                            1.772484,  -1.772484, -1.772484],
                            [0.,  1.,  0.,
                            1.,  0.,  0., 0.,
                            0.05,  0.2,  0.05,
                            1.,
                            -1.772484, 1.772484, -1.772484],
                            [0.,  0.,  1.,
                            1.,  0.,  0., 0.,
                            0.05,  0.05,  0.2,
                            1.,
                            -1.772484, -1.772484,  1.772484]
                            ], dtype=np.float32)
    app = QApplication([])
    gs_item = GaussianItem()
    grid_item = GridItem()

    items = {"gs": gs_item, "grid": grid_item}
    
    viewer = Viewer(items)
    viewer.items["gs"].setData(gs_data=gs_data)
    viewer.show()
    app.exec_()
