from gausplat import *


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

    depth = pc[:, 2]

    # step2. Calcuate the 3d Gaussian.
    cov3d = compute_cov_3d(gs['scale'], gs['rot'])

    # step3. Project the 3D Gaussian to 2d image as a 2d Gaussian.
    cov2d = compute_cov_2d(pc, camera.K, cov3d, camera.Rcw)

    # step4. get color info
    ray_dir = pw[:, :3] - camera.cam_center
    ray_dir /= np.linalg.norm(ray_dir, axis=1)[:, np.newaxis]
    color = sh2color(gs['sh'], ray_dir)

    # step5. Blend the 2d Gaussian to image
    image = splat(camera.height, camera.width, u, cov2d, gs['alpha'], depth, color)
    plt.imshow(image)
    # from PIL import Image
    # pil_img = Image.fromarray((np.clip(image, 0, 1)*255).astype(np.uint8))
    # print(pil_img.mode)
    # pil_img.save('test.png')

    plt.show()
