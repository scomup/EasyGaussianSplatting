import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from gsplat.pytorch_ssim import gau_loss
from gsplat.gau_io import *
from gsplat.gausplat_dataset import *
from gsnet import GSNet, logit
from random import randint
import time


if __name__ == "__main__":
    path = '/home/liu/bag/gaussian-splatting/tandt/train'
    gs_set = GSplatDataset(path)

    gsnet = GSNet

    ply_fn = "/home/liu/workspace/gaussian-splatting/output/train/point_cloud/iteration_10/point_cloud.ply"
    gs = load_ply(ply_fn)
    ## gs = gs_set.gs

    pws = torch.from_numpy(gs['pw']).type(
        torch.float32).to('cuda').requires_grad_()
    rots_raw = torch.from_numpy(gs['rot']).type(
    # the unactivated scales
        torch.float32).to('cuda').requires_grad_()
    scales_raw = torch.log(torch.from_numpy(gs['scale']).type(
        torch.float32).to('cuda')).requires_grad_()
    # the unactivated alphas
    alphas_raw = logit(torch.from_numpy(gs['alpha']).type(
        torch.float32).to('cuda')).requires_grad_()

    shs = torch.from_numpy(gs['sh']).type(
        torch.float32).to('cuda').requires_grad_()

    l = [
        {'params': [rots_raw], 'lr': 0.001, "name": "rot"},
        {'params': [scales_raw], 'lr': 0.005, "name": "scale"},
        {'params': [shs], 'lr': 0.001, "name": "sh"},
        {'params': [alphas_raw], 'lr': 0.001, "name": "alpha"},
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
            image = gsnet.apply(rots, scales, shs, alphas, pws, cam)
            loss = gau_loss(image, image_gt)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            avg_loss += loss.item()
            if (i == 0 and epoch % 10 == 0):
                im_cpu = image.to('cpu').detach().permute(1, 2, 0).numpy()
                im.set_data(im_cpu)
                fig.canvas.flush_events()
                plt.pause(0.1)
        avg_loss = avg_loss / n
        print("epoch:%d avg_loss:%f" % (epoch, avg_loss))
        #if (epoch % 10 == 0):
        #    save_torch_params("epoch%04d.npy" % epoch, rots, scales, shs, alphas, pws)
