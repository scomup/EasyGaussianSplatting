from gausplat import *


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", help="the ply path")
    args = parser.parse_args()

    if args.ply:
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

    pws = gs['pos']

    # step1. Transform pw to camera frame,
    # and project it to iamge.
    us, pcs = project(pws, camera.Rcw, camera.tcw, camera.K)

    depths = pcs[:, 2]

    # step2. Calcuate the 3d Gaussian.
    cov3ds = compute_cov_3d(gs['scale'], gs['rot'])

    # step3. Project the 3D Gaussian to 2d image as a 2d Gaussian.
    cov2ds = compute_cov_2d(pcs, camera.K, cov3ds, camera.Rcw)

    # step4. get color info
    colors = sh2color(gs['sh'], pws, camera.cam_center)

    # step5. Blend the 2d Gaussian to image
    cinv2ds, areas = inverse_cov2d(cov2ds)

    image = splat(camera.height, camera.width, us, cinv2ds, gs['alpha'], depths, colors, areas)
    plt.imshow(image)
    # from PIL import Image
    # pil_img = Image.fromarray((np.clip(image, 0, 1)*255).astype(np.uint8))
    # print(pil_img.mode)
    # pil_img.save('test.png')

    plt.show()
