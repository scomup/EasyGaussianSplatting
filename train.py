import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from gsplat.pytorch_ssim import gau_loss
from gsplat.gau_io import *
from gsplat.gausplat_dataset import *
from gsmodel import *


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

    gs_params = get_gs_params(gs_set.gs)

    l = [
        {'params': [gs_params['pws']], 'lr': 0.001, "name": "pws"},
        {'params': [gs_params['low_shs']], 'lr': 0.001, "name": "low_shs"},
        {'params': [gs_params['high_shs']], 'lr': 0.001/20, "name": "high_shs"},
        {'params': [gs_params['alphas_raw']], 'lr': 0.05, "name": "alphas_raw"},
        {'params': [gs_params['scales_raw']], 'lr': 0.005, "name": "scales_raw"},
        {'params': [gs_params['rots_raw']], 'lr': 0.001, "name": "rots_raw"},
    ]

    optimizer = optim.Adam(l, lr=0.000, eps=1e-15)

    cam0, _ = gs_set[0]
    fig, ax = plt.subplots()
    array = np.zeros(shape=(cam0.height, cam0.width, 3), dtype=np.uint8)
    im = ax.imshow(array)

    epochs = 100
    n = len(gs_set)
    model = GSModel(gs_set.sence_size, len(gs_set) * epochs)

    for epoch in range(1, epochs):
        idxs = np.arange(n)
        np.random.shuffle(idxs)
        avg_loss = 0
        for i in idxs:
            cam, image_gt = gs_set[i]

            image = model(*gs_params.values(), cam)
            loss = gau_loss(image, image_gt)
            loss.backward()
    
            model.update_density_info()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    
            model.update_pws_lr(optimizer)
            avg_loss += loss.item()
    
            if (i == 0):
                im.set_data(np.clip(image.detach().permute(1, 2, 0).to('cpu').numpy(), 0, 1))
                plt.pause(0.1)

        avg_loss = avg_loss / n
        print("epoch:%d avg_loss:%f" % (epoch, avg_loss))
        with torch.no_grad():
            if (epoch <= 50):
                if (epoch % 5 == 0):
                    model.update_gaussian_density(gs_params, optimizer)
                if (epoch % 15 == 0):
                    print("reset aplha")
                    model.reset_alpha(gs_params, optimizer)
            # save data.
            if (epoch % 10 == 0):
                fn = "data/epoch%04d.npy" % epoch
                save_gs_params(fn, gs_params)
                print("trained data is saved to %s" % fn)
    
    save_gs_params('data/final.npy', gs_params)
    print("Training is finished.")