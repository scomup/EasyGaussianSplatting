from gsplat.gausplat import *
from gsplat.gau_io import *


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gs", help="the input 3d gaussians path")
    args = parser.parse_args()

    if args.gs:
        ply_fn = args.gs
        print("Try to load %s ..." % ply_fn)
        gs = load_gs(ply_fn)
    else:
        print("not gaussians file.")
        gs = get_example_gs()

    # Camera info
    tcw = np.array([1.03796196, 0.42017467, 4.67804612])
    Rcw = np.array([[0.89699204,  0.06525223,  0.43720409],
                    [-0.04508268,  0.99739184, -0.05636552],
                    [-0.43974177,  0.03084909,  0.89759429]]).T

    width = int(979)  # 1957  # 979
    height = int(546)  # 1091  # 546

    fx = 581.6273640151177
    fy = 578.140202494143
    cx = width / 2
    cy = height / 2

    twc = np.linalg.inv(Rcw) @ (-tcw)

    fig, ax = plt.subplots()
    array = np.zeros(shape=(height, width, 3), dtype=np.uint8)
    im = ax.imshow(array)

    pws = gs['pw']

    # step1. Transform pw to camera frame,
    # and project it to iamge.
    us, pcs = project(pws, Rcw, tcw, fx, fy, cx, cy)

    depths = pcs[:, 2]

    # step2. Calcuate the 3d Gaussian.
    cov3ds = compute_cov_3d(gs['scale'], gs['rot'])

    # step3. Project the 3D Gaussian to 2d image as a 2d Gaussian.
    cov2ds = compute_cov_2d(pcs, fx, fy, width, height, cov3ds, Rcw)

    # step4. get color info
    colors = sh2color(gs['sh'], pws, twc)

    # step5. Blend the 2d Gaussian to image
    cinv2ds, areas = inverse_cov2d(cov2ds)

    splat(height, width, us, cinv2ds, gs['alpha'],
          depths, colors, areas, im)

    # from PIL import Image
    # pil_img = Image.fromarray((np.clip(image, 0, 1)*255).astype(np.uint8))
    # print(pil_img.mode)
    # pil_img.save('test.png')
    #  plt.imshow(image)
    plt.show()
