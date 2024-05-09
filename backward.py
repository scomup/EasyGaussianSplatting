from gaussian_splatting import *


def splat_test(H, W, u, cov2d, alpha, depth, color):
    import torch
    import simple_gaussian_reasterization as sgr
    u = torch.from_numpy(u).type(torch.float32).to('cuda')
    cov2d = torch.from_numpy(cov2d).type(torch.float32).to('cuda')
    alpha = torch.from_numpy(alpha).type(torch.float32).to('cuda')
    depth = torch.from_numpy(depth).type(torch.float32).to('cuda')
    color = torch.from_numpy(color).type(torch.float32).to('cuda')
    res = sgr.forward(H, W, u, cov2d, alpha, depth, color)

    contrib = res[1]
    final_tau = res[2]
    patch_offset_per_tile = res[3]
    gs_id_per_patch = res[4]
    import torch.nn as nn
    criterion = nn.L1Loss()

    image = res[0].requires_grad_(True)
    image_gt = torch.zeros([3, H, W], dtype=torch.float32).to('cuda')
    loss = criterion(image, image_gt)
    loss.backward()
    dloss_dgammas = image.grad
    # dloss_dgammas = torch.ones([3, H, W], dtype=torch.float32).to('cuda')

    jacobians = sgr.backward(H, W, u, cov2d, alpha, depth, color,
                             contrib, final_tau, patch_offset_per_tile,
                             gs_id_per_patch, dloss_dgammas)
    print("dloss_du:\n", jacobians[0])
    print("dloss_dcov2d:\n", jacobians[1])
    print("dloss_dalpha:\n", jacobians[2])
    print("dloss_dcolor:\n", jacobians[3])
    res[0].detach().to('cpu').numpy().transpose(1, 2, 0)
    return contrib.detach().to('cpu').numpy()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", help="the ply path")
    args = parser.parse_args()

    if args.ply:
        # ply_fn = "/home/liu/workspace/gaussian-splatting/output/test/point_cloud/iteration_30000/point_cloud.ply"
        ply_fn = args.ply
        print("Try to load %s ..." % ply_fn)
        gs = load_ply(ply_fn)
    else:
        print("not fly file.")
        # exit(0)
        # ply_fn = "/home/liu/workspace/gaussian-splatting/output/test/point_cloud/iteration_30000/point_cloud.ply"
        # gs = load_ply(ply_fn)

    gs_data = np.array([[0.,  0.,  0.,  # xyz
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
                        ], dtype=np.float32)

    dtypes = [('pos', '<f4', (3,)),
              ('rot', '<f4', (4,)),
              ('scale', '<f4', (3,)),
              ('alpha', '<f4'),
              ('sh', '<f4', (3,))]

    gs = np.frombuffer(gs_data.tobytes(), dtype=dtypes)

    # Camera info
    tcw = np.array([1.03796196, 0.42017467, 4.67804612])
    Rcw = np.array([[0.89699204,  0.06525223,  0.43720409],
                    [-0.04508268,  0.99739184, -0.05636552],
                    [-0.43974177,  0.03084909,  0.89759429]]).T

    # W = int(979)  # 1957  # 979
    # H = int(546)  # 1091  # 546
    # focal_x = 1163.2547280302354/2.
    # focal_y = 1156.280404988286/2.

    W = int(32)  # 1957  # 979
    H = int(16)  # 1091  # 546
    focal_x = 16
    focal_y = 16

    K = np.array([[focal_x, 0, W/2.],
                  [0, focal_y, H/2.],
                  [0, 0, 1.]])

    camera = Camera(id=0, width=W, height=H, K=K, Rcw=Rcw, tcw=tcw)

    pw = gs['pos']

    # step1. Transform pw to camera frame,
    # and project it to iamge.
    u, pc = project(pw, camera.Rcw, camera.tcw, camera.K)

    depth = pc[:, 2]

    # step2. Calcuate the 3d Gaussian.
    cov3d = compute_cov_3d(gs['scale'], gs['rot'])

    # step3. Project the 3D Gaussian to 2d image as a 2d Gaussian.
    cov2d = compute_cov_2d(pc, camera.focal_x, camera.focal_y, cov3d, camera.Rcw)

    # step4. get color info
    ray_dir = pw[:, :3] - camera.cam_center
    ray_dir /= np.linalg.norm(ray_dir, axis=1)[:, np.newaxis]
    color = sh2color(gs['sh'], ray_dir)

    # step5. Blend the 2d Gaussian to image
    image = splat_test(camera.height, camera.width, u, cov2d, gs['alpha'], depth, color)

    plt.imshow(image)

    plt.show()
