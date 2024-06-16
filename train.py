import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from gsplat.pytorch_ssim import gau_loss
from gsplat.gau_io import *
from gsplat.gausplat_dataset import *
from gsmodel import *
from random import randint
import time
import gsplatcu as gsc
from gsplat.utils import get_expon_lr_func


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
    # path = "/home/liu/bag/colmap"
    path = "/home/liu/bag/gaussian-splatting/tandt/train"
    gs_set = GSplatDataset(path, resize_rate=1)

    gs = gs_set.gs

    pws = torch.from_numpy(gs['pw']).type(
        torch.float32).to('cuda').requires_grad_()
    rots_raw = torch.from_numpy(gs['rot']).type(
        # the unactivated scales
        torch.float32).to('cuda').requires_grad_()
    scales_raw = get_scales_raw(torch.from_numpy(gs['scale']).type(
        torch.float32).to('cuda')).requires_grad_()
    # the unactivated alphas
    alphas_raw = get_alphas_raw(torch.from_numpy(gs['alpha'][:, np.newaxis]).type(
        torch.float32).to('cuda')).requires_grad_()

    low_shs = torch.from_numpy(gs['sh']).type(
        torch.float32).to('cuda').requires_grad_()

    high_shs = torch.ones_like(low_shs).repeat(1, 15) * 0.001
    high_shs = high_shs.requires_grad_()

    l = [
        {'params': [pws], 'lr': 0.001, "name": "pws"},
        {'params': [low_shs], 'lr': 0.001, "name": "low_shs"},
        {'params': [high_shs], 'lr': 0.001/20, "name": "high_shs"},
        {'params': [alphas_raw], 'lr': 0.05, "name": "alphas_raw"},
        {'params': [scales_raw], 'lr': 0.005, "name": "scales_raw"},
        {'params': [rots_raw], 'lr': 0.001, "name": "rots_raw"},
    ]


    gs_params = {"pws": pws, "low_shs": low_shs, "high_shs": high_shs,
                 "alphas_raw": alphas_raw, "scales_raw": scales_raw, "rots_raw": rots_raw}

    optimizer = optim.Adam(l, lr=0.000, eps=1e-15)

    model = GSModel(gs_set.sence_size)

    cam0, _ = gs_set[0]
    fig, ax = plt.subplots()
    array = np.zeros(shape=(cam0.height, cam0.width, 3), dtype=np.uint8)
    im = ax.imshow(array)

    n_epochs = 100
    n = len(gs_set)
    iteration = 0

    pws_lr_scheduler = get_expon_lr_func(lr_init=1e-4 * gs_set.sence_size,
                                        lr_final=1e-6 * gs_set.sence_size,
                                        lr_delay_mult=0.01,
                                        max_steps=n_epochs * n)


    for epoch in range(1, n_epochs):
        idxs = np.arange(n)
        np.random.shuffle(idxs)
        avg_loss = 0
        # us is not involved in the forward,
        # but in order to obtain the dloss_dus, we need to pass it to GSModel.
        us = torch.zeros([gs_params['pws'].shape[0], 2], dtype=torch.float32,
                         device='cuda', requires_grad=True)

        for i in idxs:
            cam, image_gt = gs_set[i]
            image, areas = model(*gs_params.values(), us, cam)
            loss = gau_loss(image, image_gt)
            loss.backward()
            model.update_density_info(us.grad, areas)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pws_lr = pws_lr_scheduler(iteration)
            iteration += 1
            pws_param = list(filter(lambda x: x["name"] == "pws", optimizer.param_groups))[0]
            pws_param['lr'] = pws_lr

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
        with torch.no_grad():
            if (epoch <= 50):
                if (epoch % 5 == 0):
                    model.update_gaussian_density(gs_params, optimizer)
                if (epoch % 15 == 0):
                    print("reset aplha")
                    model.reset_alpha(gs_params, optimizer)
            if (epoch % 10 == 0):
                fn = "data/epoch%04d.npy" % epoch
                save_gs_params(fn, gs_params)
                print("trained data is saved to %s" % fn)
