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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="the path of dataset")
    args = parser.parse_args()

    if args.path:
        print("Try to training %s ..." % args.path)
        gs_set = GSplatDataset(args.path)
    else:
        print("not path of dataset.")
        exit(0)

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

    cam0, _ = gs_set[0]
    fig, ax = plt.subplots()
    array = np.zeros(shape=(cam0.height, cam0.width, 3), dtype=np.uint8)
    im = ax.imshow(array)

    n_epochs = 100
    n = len(gs_set)
    n = 1
    for epoch in range(n_epochs):
        idxs = np.arange(n)
        np.random.shuffle(idxs)
        avg_loss = 0
        for i in idxs:
            cam, image_gt = gs_set[i]
            # Limit the value of alphas: 0 < alphas < 1
            alphas = torch.sigmoid(alphas_raw)
            # Limit the value of scales > 0
            scales = torch.exp(scales_raw)
            # Limit the value of rot, normal of rots is 1
            rots = torch.nn.functional.normalize(rots_raw)

            image = GSModel.apply(
                pws,
                shs,
                alphas,
                scales,
                rots,
                cam,
            )

            loss = gau_loss(image, image_gt)
            loss.backward()
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
        if (epoch % 10 == 0):
            fn = "data/epoch%04d.npy" % epoch
            print("trained data is saved to %s" % fn)
            save_torch_params(fn, rots, scales, shs, alphas, pws)
