import torch
import gsplatcu as gsc
import numpy as np
import matplotlib.pyplot as plt
from gsplat.gau_io import *


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gs", help="the gs path")
    args = parser.parse_args()

    if args.gs:
        print("Try to load %s ..." % args.gs)
        gs = load_gs(args.gs)
    else:
        print("not gs file.")
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

        dtypes = [('pw', '<f4', (3,)),
                  ('rot', '<f4', (4,)),
                  ('scale', '<f4', (3,)),
                  ('alpha', '<f4'),
                  ('sh', '<f4', (3,))]

        gs = np.frombuffer(gs_data.tobytes(), dtype=dtypes)

    # ply_fn = "/home/liu/workspace/gaussian-splatting/output/test/point_cloud/iteration_30000/point_cloud.ply"
    # gs = load_ply(ply_fn)

    # Camera info
    tcw = np.array([1.03796196, 0.42017467, 4.67804612])
    Rcw = np.array([[0.89699204,  0.06525223,  0.43720409],
                    [-0.04508268,  0.99739184, -0.05636552],
                    [-0.43974177,  0.03084909,  0.89759429]]).T

    width = int(979)  # 1957
    height = int(546)  # 1091

    focal_x = 581.6273640151177
    focal_y = 578.140202494143
    center_x = width / 2
    center_y = height / 2

    pws = torch.from_numpy(gs['pw']).type(torch.float32).to('cuda')
    rots = torch.from_numpy(gs['rot']).type(torch.float32).to('cuda')
    scales = torch.from_numpy(gs['scale']).type(torch.float32).to('cuda')
    alphas = torch.from_numpy(gs['alpha']).type(torch.float32).to('cuda')
    shs = torch.from_numpy(gs['sh']).type(torch.float32).to('cuda')
    Rcw = torch.from_numpy(Rcw).type(torch.float32).to('cuda')
    tcw = torch.from_numpy(tcw).type(torch.float32).to('cuda')
    twc = torch.linalg.inv(Rcw)@(-tcw)

    # step1. Transform pw to camera frame,
    # and project it to iamge.
    us, pcs, depths = gsc.project(pws, Rcw, tcw, focal_x, focal_y, center_x, center_y, False)

    # step2. Calcuate the 3d Gaussian.
    cov3ds = gsc.computeCov3D(rots, scales, depths, False)[0]

    # step3. Calcuate the 2d Gaussian.
    cov2ds = gsc.computeCov2D(cov3ds, pcs, depths, Rcw, focal_x, focal_y, height, width, False)[0]

    # step4. get color info
    colors = gsc.sh2Color(shs, pws, twc, False)[0]

    # step5. Blend the 2d Gaussian to image
    cinv2ds, areas = gsc.inverseCov2D(cov2ds, depths, False)
    image = gsc.splat(height, width, us, cinv2ds, alphas, depths, colors, areas)[0]
    image = image.to('cpu').numpy()

    plt.imshow(image.transpose([1, 2, 0]))
    plt.show()
