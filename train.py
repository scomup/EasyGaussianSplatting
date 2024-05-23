import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
from gsplat.pytorch_ssim import gau_loss
from gsplat.read_ply import *
from gsplat.gausplat_dataset import *
from gsnet import GSNet


if __name__ == "__main__":
    path = '/home/liu/bag/gaussian-splatting/tandt/train'
    gs_dataset = GSplatDataset(path, device='cuda')
    cam0 = gs_dataset[0]
    gsnet = GSNet
    gs = gs_dataset.gs

    image_gt = cam0.image.requires_grad_()

    ply_fn = "/home/liu/workspace/gaussian-splatting/output/train/point_cloud/iteration_10/point_cloud.ply"
    gs = load_ply(ply_fn)

    pws = torch.from_numpy(gs['pw']).type(torch.float32).to('cuda').requires_grad_()
    rots = torch.from_numpy(gs['rot']).type(torch.float32).to('cuda').requires_grad_()
    scales = torch.from_numpy(gs['scale']).type(torch.float32).to('cuda').requires_grad_()
    alphas = torch.from_numpy(gs['alpha']).type(torch.float32).to('cuda').requires_grad_()
    shs = torch.from_numpy(gs['sh']).type(torch.float32).to('cuda').requires_grad_()

    optimizer = optim.Adam(
        [rots, scales, shs, alphas, pws], lr=0.001, eps=1e-15)

    fig, ax = plt.subplots()
    array = np.zeros(shape=(cam0.height, cam0.width, 3), dtype=np.uint8)
    im = ax.imshow(array)

    for i in range(100):
        image = gsnet.apply(rots, scales, shs, alphas, pws, cam0)
        loss = gau_loss(image, image_gt)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if (i % 1 == 0):
            print("step:%d loss:%f" % (i, loss.item()))
            im_cpu = image.to('cpu').detach().permute(1, 2, 0).numpy()
            im.set_data(im_cpu)
            fig.canvas.flush_events()
            plt.pause(1)
    plt.show()
