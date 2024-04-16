#!/usr/bin/env python3

from viewer import *
import numpy as np
from custom_items import *

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from sh_demo import sh2color


sh = np.array([[-0.8756, -0.6117, -0.1774],
               [0.1267,  0.0838,  0.0142],
               [0.1620,  0.1117, -0.0665],
               [0.1706,  0.1166, -0.0173],
               [0.0614,  0.0457,  0.0524],
               [0.0772,  0.0486, -0.0343],
               [0.2367,  0.2127,  0.1651],
               [0.0905,  0.0642,  0.0164],
               [-0.0043, -0.0020,  0.0506],
               [0.1112,  0.0699, -0.0328],
               [0.1567,  0.1012, -0.0047],
               [0.0430,  0.0315, -0.0233],
               [-0.0987, -0.0805, -0.0461],
               [-0.0660, -0.0421,  0.0191],
               [-0.0450, -0.0263,  0.0552],
               [0.0394,  0.0264,  0.0118],
               [0.0881,  0.0545, -0.0780],
               [-0.0088,  0.0032, -0.0070],
               [0.0040,  0.0014, -0.0376],
               [-0.1464, -0.1240, -0.0949],
               [0.3520,  0.3167,  0.2688],
               [-0.1495, -0.1115, -0.0557],
               [-0.1255, -0.0845,  0.0146],
               [0.0466,  0.0307, -0.0380],
               [-0.0069,  0.0048,  0.0165],
               [0.0418,  0.0253, -0.0354],
               [-0.0477, -0.0230, -0.0013],
               [0.0037,  0.0041, -0.0197],
               [-0.0760, -0.0521, -0.0327],
               [0.0422,  0.0287,  0.0154],
               [0.0401,  0.0138, -0.0881],
               [0.0605,  0.0490,  0.0426],
               [-0.0519, -0.0320,  0.0086],
               [0.0196,  0.0088, -0.0307],
               [0.1187,  0.0742, -0.0387],
               [0.0025,  0.0038,  0.0028]], dtype=np.float32)

if __name__ == '__main__':
    app = QApplication([])
    gt = SphereItem()
    sh1 = SphereItem()
    c1, _ = sh2color(sh, sh1.vertices, dim=4)
    sh2 = SphereItem()
    c2, _ = sh2color(sh, sh2.vertices, dim=9)
    sh3 = SphereItem()
    c3, _ = sh2color(sh, sh3.vertices, dim=16)
    sh4 = SphereItem()
    c4, _ = sh2color(sh, sh4.vertices, dim=25)

    gt.set_colors_from_image('imgs/Solarsystemscope_texture_8k_earth_daymap.jpg')
    sh1.set_colors(c1.T)
    sh2.set_colors(c2.T)
    sh3.set_colors(c3.T)
    sh4.set_colors(c4.T)

    s = 3.
    a1 = np.eye(4)
    a1[0, 3] = 1 * s
    sh1.setTransform(a1)

    a2 = np.eye(4)
    a2[0, 3] = 2 * s
    sh2.setTransform(a2)

    a3 = np.eye(4)
    a3[0, 3] = 3 * s
    sh3.setTransform(a3)

    a4 = np.eye(4)
    a4[0, 3] = 4 * s
    sh4.setTransform(a4)

    items = {"gt": gt, "sh1": sh1, "sh2": sh2, "sh3": sh3, "sh4": sh4}
    viewer = Viewer(items)
    viewer.show()
    app.exec_()
