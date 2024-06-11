import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from gsplat.pytorch_ssim import gau_loss
from gsplat.gau_io import *
from gsplat.gausplat_dataset import *
from gsmodel import GSModel, logit
from random import randint
import time
import gsplatcu as gsc


def rainbow(scalars, scalar_min=0, scalar_max=255):
    range = scalar_max - scalar_min
    values = 1.0 - (scalars - scalar_min) / range
    # values = (scalars - scalar_min) / range  # using inverted color
    colors = torch.zeros([scalars.shape[0], 3], dtype=torch.float32, device='cuda')
    values = torch.clip(values, 0, 1)

    h = values * 5.0 + 1.0
    i = torch.floor(h).to(torch.int32)
    f = h - i
    f[torch.logical_not(i % 2)] = 1 - f[torch.logical_not(i % 2)]
    n = 1 - f

    # idx = i <= 1
    colors[i <= 1, 0] = n[i <= 1]
    colors[i <= 1, 1] = 0
    colors[i <= 1, 2] = 1

    colors[i == 2, 0] = 0
    colors[i == 2, 1] = n[i == 2]
    colors[i == 2, 2] = 1

    colors[i == 3, 0] = 0
    colors[i == 3, 1] = 1
    colors[i == 3, 2] = n[i == 3]

    colors[i == 4, 0] = n[i == 4]
    colors[i == 4, 1] = 1
    colors[i == 4, 2] = 0

    colors[i >= 5, 0] = 1
    colors[i >= 5, 1] = n[i >= 5]
    colors[i >= 5, 2] = 0
    shs = (colors - 0.5) / 0.28209479177387814
    return shs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="the path of dataset")
    args = parser.parse_args()

    # if args.path:
    #     print("Try to training %s ..." % args.path)
    #     gs_set = GSplatDataset(args.path)
    # else:
    #     print("not path of dataset.")
    #     exit(0)
    path = "/home/liu/bag/colmap"
    # path = "/home/liu/bag/gaussian-splatting/tandt/train"
    gs_set = GSplatDataset(path, resize_rate=1)

    gs = gs_set.gs

    pws = torch.from_numpy(gs['pw']).type(
        torch.float32).to('cuda').requires_grad_()
    rots_raw = torch.from_numpy(gs['rot']).type(
        # the unactivated scales
        torch.float32).to('cuda').requires_grad_()
    scales_raw = torch.log(torch.from_numpy(gs['scale']).type(
        torch.float32).to('cuda')).requires_grad_()
    # the unactivated alphas
    alphas_raw = logit(torch.from_numpy(gs['alpha'][:, np.newaxis]).type(
        torch.float32).to('cuda')).requires_grad_()

    shs = torch.from_numpy(gs['sh']).type(
        torch.float32).to('cuda').requires_grad_()

    l = [
        {'params': [rots_raw], 'lr': 0.001, "name": "rot"},
        {'params': [scales_raw], 'lr': 0.005, "name": "scale"},
        {'params': [shs], 'lr': 0.001, "name": "sh"},
        {'params': [alphas_raw], 'lr': 0.05, "name": "alpha"},
        {'params': [pws], 'lr': 0.001, "name": "pw"}
    ]

    optimizer = optim.Adam(l, lr=0.000, eps=1e-15)

    model = GSModel()

    cam0, _ = gs_set[0]
    fig, ax = plt.subplots()
    array = np.zeros(shape=(cam0.height, cam0.width, 3), dtype=np.uint8)
    im = ax.imshow(array)

    n_epochs = 1000
    n = len(gs_set)
    # n = 1
    for epoch in range(n_epochs):
        idxs = np.arange(n)
        np.random.shuffle(idxs)
        avg_loss = 0
        for i in idxs:
            cam, image_gt = gs_set[i]
            # us is not involved in the forward,
            # but in order to obtain the dloss_dus, we need to pass it to GSModel.
            us = torch.zeros([pws.shape[0], 2],
                             dtype=torch.float32, device='cuda', requires_grad=True)

            image, areas = model(
                pws,
                shs,
                alphas_raw,
                scales_raw,
                rots_raw,
                us,
                cam)

            loss = gau_loss(image, image_gt)
            loss.backward()
            model.update_density_info(us.grad, areas)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            avg_loss += loss.item()
            if (i == 0):
                im_cpu = image.to('cpu').detach().permute(1, 2, 0).numpy()
                im_cpu = np.clip(im_cpu, 0, 1)
                im.set_data(im_cpu)
                fig.canvas.flush_events()
                plt.pause(0.1)
                # plt.show()
        avg_loss = avg_loss / n
        print("epoch:%d avg_loss:%f" % (epoch, avg_loss))
        # save data.
        if (epoch % 10 == 5):
            fn = "data/epoch%04d.npy" % epoch
            print("trained data is saved to %s" % fn)
            # shs = torch.zeros_like(shs)
            with torch.no_grad():
                grad = torch.norm(model.dloss_dus_accum) / model.cunt
                shs = rainbow(grad, 0, 0.0002 / 1000)
            save_torch_params(fn, rots_raw, scales_raw, shs, grad, pws)
            exit()
