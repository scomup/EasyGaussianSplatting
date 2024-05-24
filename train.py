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
    # gs = gs_set.gs

    pws = torch.from_numpy(gs['pw']).type(
        torch.float32).to('cuda').requires_grad_()
    rots = torch.from_numpy(gs['rot']).type(
        torch.float32).to('cuda').requires_grad_()
    scales = torch.log(torch.from_numpy(gs['scale']).type(
        torch.float32).to('cuda')).requires_grad_()
    alphas = logit(torch.from_numpy(gs['alpha']).type(
        torch.float32).to('cuda')).requires_grad_()
    shs = torch.from_numpy(gs['sh']).type(
        torch.float32).to('cuda').requires_grad_()

    l = [
        {'params': [rots], 'lr': 0.001, "name": "rot"},
        {'params': [scales], 'lr': 0.005, "name": "scale"},
        {'params': [shs], 'lr': 0.001, "name": "sh"},
        {'params': [alphas], 'lr': 0.001, "name": "alpha"},
        {'params': [pws], 'lr': 0.001, "name": "pw"}
    ]

    # l = [rots, scales, shs, alphas, pws]
    optimizer = optim.Adam(l, lr=0.001, eps=1e-15)

    cam0, _ = gs_set[0]
    fig, ax = plt.subplots()
    array = np.zeros(shape=(cam0.height, cam0.width, 3), dtype=np.uint8)
    im = ax.imshow(array)

    n_epochs = 1
    for epoch in range(n_epochs):
        idxs = np.arange(len(gs_set))
        np.random.shuffle(idxs)
        avg_loss = 0
        for i in idxs:
            cam, image_gt = gs_set[i]
            image = gsnet.apply(rots, scales, shs, alphas, pws, cam)
            loss = gau_loss(image, image_gt)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            avg_loss += loss.item()
            if (i == 0):
                im_cpu = image.to('cpu').detach().permute(1, 2, 0).numpy()
                im.set_data(im_cpu)
                fig.canvas.flush_events()
                plt.pause(0.1)
        avg_loss = avg_loss / len(gs_set)
        print("epoch:%d avg_loss:%f" % (epoch, avg_loss))
        save_torch_params("epoch_%04d.npy"%epoch, rots, scales, shs, alphas, pws, cam)
