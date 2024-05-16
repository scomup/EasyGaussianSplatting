from gaussian_splatting import *
import torch
import simple_gaussian_reasterization as sgr


def project_gpu(pw, Rcw, tcw, K):
    pw = torch.from_numpy(pw).type(torch.float32).to('cuda')
    Rcw = torch.from_numpy(Rcw).type(torch.float32).to('cuda')
    tcw = torch.from_numpy(tcw).type(torch.float32).to('cuda')
    u, pc = sgr.project(pw, Rcw, tcw, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    return u.to('cpu').numpy(), pc.to('cpu').numpy()


def compute_cov_3d_gpu(scale, rot):
    scale = torch.from_numpy(scale).type(torch.float32).to('cuda')
    rot = torch.from_numpy(rot).type(torch.float32).to('cuda')
    res = sgr.computeCov3D(rot, scale)
    return res[0].to('cpu').numpy()


def compute_cov_2d_gpu(pc, K, cov3d, Rcw):
    pc = torch.from_numpy(pc).type(torch.float32).to('cuda')
    cov3d = torch.from_numpy(cov3d).type(torch.float32).to('cuda')
    Rcw = torch.from_numpy(Rcw).type(torch.float32).to('cuda')
    focal_x = K[0, 0]
    focal_y = K[1, 1]
    res = sgr.computeCov2D(cov3d, pc, Rcw, focal_x, focal_y)
    return res[0].to('cpu').numpy()


def splat_gpu(height, width, u, cov2d, alpha, depth, color):
    u = torch.from_numpy(u).type(torch.float32).to('cuda')
    cov2d = torch.from_numpy(cov2d).type(torch.float32).to('cuda')
    alpha = torch.from_numpy(alpha).type(torch.float32).to('cuda')
    depth = torch.from_numpy(depth).type(torch.float32).to('cuda')
    color = torch.from_numpy(color).type(torch.float32).to('cuda')
    res = sgr.forward(height, width, u, cov2d, alpha, depth, color)
    res_cpu = []
    for r in res:
        res_cpu.append(r.to('cpu').numpy())
    res_cpu[0] = res_cpu[0].transpose(1, 2, 0)
    return res_cpu


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

        dtypes = [('pos', '<f4', (3,)),
                  ('rot', '<f4', (4,)),
                  ('scale', '<f4', (3,)),
                  ('alpha', '<f4'),
                  ('sh', '<f4', (3,))]

        gs = np.frombuffer(gs_data.tobytes(), dtype=dtypes)

    ply_fn = "/home/liu/workspace/gaussian-splatting/output/test/point_cloud/iteration_30000/point_cloud.ply"
    gs = load_ply(ply_fn)

    # Camera info
    tcw = np.array([1.03796196, 0.42017467, 4.67804612])
    Rcw = np.array([[0.89699204,  0.06525223,  0.43720409],
                    [-0.04508268,  0.99739184, -0.05636552],
                    [-0.43974177,  0.03084909,  0.89759429]]).T

    width = int(979)  # 1957  # 979
    height = int(546)  # 1091  # 546

    focal_x = 581.6273640151177
    focal_y = 578.140202494143
    K = np.array([[focal_x, 0, width / 2],
                  [0, focal_y, height / 2],
                  [0, 0, 1.]])

    camera = Camera(id=0, width=width, height=height, K=K, Rcw=Rcw, tcw=tcw)

    pw = gs['pos']

    # step1. Transform pw to camera frame,
    # and project it to iamge.
    u, pc = project(pw, camera.Rcw, camera.tcw, camera.K)
    u_gpu, pc_gpu = project_gpu(pw, camera.Rcw, camera.tcw, camera.K)
    print(np.max(np.abs(u - u_gpu)))
    print(np.max(np.abs(pc - pc_gpu)))

    depth = pc[:, 2]

    # step2. Calcuate the 3d Gaussian.
    cov3d = compute_cov_3d(gs['scale'], gs['rot'])
    cov3d_gpu = compute_cov_3d_gpu(gs['scale'], gs['rot'])
    print(np.max(np.abs(cov3d - cov3d_gpu)))

    cov2d = compute_cov_2d(pc, camera.K, cov3d, camera.Rcw)
    cov2d_gpu = compute_cov_2d_gpu(pc, camera.K, cov3d, camera.Rcw)
    print(cov2d)
    print(cov2d_gpu)
    print(np.max(np.abs(cov2d - cov2d_gpu)))
    exit()

    # step3. Project the 3D Gaussian to 2d image as a 2d Gaussian.
    cov2d = compute_cov_2d(pc, camera.K, cov3d, camera.Rcw)

    # step4. get color info
    ray_dir = pw[:, :3] - camera.cam_center
    ray_dir /= np.linalg.norm(ray_dir, axis=1)[:, np.newaxis]
    color = sh2color(gs['sh'], ray_dir)

    # step5. Blend the 2d Gaussian to image
    res = splat_gpu(height, width, u, cov2d, gs['alpha'], depth, color)
    image = res[0]

    plt.imshow(image)
    # from PIL import Image
    # pil_img = Image.fromarray((np.clip(image, 0, 1)*255).astype(np.uint8))
    # print(pil_img.mode)
    # pil_img.save('test.png')

    plt.show()
