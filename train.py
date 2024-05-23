import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from gsplat.pytorch_ssim import gau_loss
from gsplat.read_ply import *
from gsplat.gausplat_dataset import *
from gsnet import GSNet
from random import randint
import time


if __name__ == "__main__":
    path = '/home/liu/bag/gaussian-splatting/tandt/train'
    gs_set = GSplatDataset(path, device='cuda')

    gsnet = GSNet

    ply_fn = "/home/liu/workspace/gaussian-splatting/output/train/point_cloud/iteration_10/point_cloud.ply"
    gs = load_ply(ply_fn)

    pws = torch.from_numpy(gs['pw']).type(torch.float32).to('cuda').requires_grad_()
    rots = torch.from_numpy(gs['rot']).type(torch.float32).to('cuda').requires_grad_()
    scales = torch.from_numpy(gs['scale']).type(torch.float32).to('cuda').requires_grad_()
    alphas = torch.from_numpy(gs['alpha']).type(torch.float32).to('cuda').requires_grad_()
    shs = torch.from_numpy(gs['sh']).type(torch.float32).to('cuda').requires_grad_()

    optimizer = optim.Adam(
        [rots, scales, shs, alphas, pws], lr=0.001, eps=1e-15)

    cam0, _ = gs_set[0]
    fig, ax = plt.subplots()
    array = np.zeros(shape=(cam0.height, cam0.width, 3), dtype=np.uint8)
    im = ax.imshow(array)

    for iteration in range(0, 10000):
        i = randint(0, len(gs_set) - 1)
        cam, image_gt = gs_set[i]
        image = gsnet.apply(rots, scales, shs, alphas, pws, cam)
        loss = gau_loss(image, image_gt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print("step:%d image: %d loss:%f" % (iteration, i, loss.item()))
        if (i == 0):
            print("step:%d loss:%f" % (iteration, loss.item()))
            im_cpu = image.to('cpu').detach().permute(1, 2, 0).numpy()
            im.set_data(im_cpu)
            fig.canvas.flush_events()
            plt.pause(0.1)
        
