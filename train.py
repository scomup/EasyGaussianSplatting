import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from gsplat.pytorch_ssim import gau_loss
from gsplat.gau_io import *
from gsplat.gausplat_dataset import *
from gsplat.gsmodel import *


torch.autograd.set_detect_anomaly(True)


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
    # path = "/home/liu/bag/gaussian-splatting/tandt/train"
    # gs_set = GSplatDataset(path, resize_rate=1)
    # gs = np.load("data/final.npy")

    training_params, adam_params = get_training_params(gs_set.gs)

    optimizer = optim.Adam(adam_params, lr=0.000, eps=1e-15)

    cam0, _ = gs_set[0]
    fig, ax = plt.subplots()
    img = ax.imshow(
        np.zeros(shape=(cam0.height, cam0.width, 3), dtype=np.uint8))
    txt = ax.text(50, 50, "", size=20, color='white')

    epochs = 100
    n = len(gs_set)
    model = GSModel(gs_set.sence_size, len(gs_set) * epochs)

    for epoch in range(epochs):
        idxs = np.arange(n)
        np.random.shuffle(idxs)
        avg_loss = 0
        for i in idxs:
            cam, image_gt = gs_set[i]

            image = model(*training_params.values(), cam)
            loss = gau_loss(image, image_gt)
            loss.backward()

            model.update_density_info()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            model.update_pws_lr(optimizer)
            avg_loss += loss.item()

            if (i == 0):
                img.set_data(np.clip(image.detach().permute(
                    1, 2, 0).to('cpu').numpy(), 0, 1))
                txt._text = "epoch %d" % epoch
                plt.pause(0.1)

        avg_loss = avg_loss / n
        print("epoch:%d avg_loss:%f" % (epoch, avg_loss))
        with torch.no_grad():
            if (epoch > 1 and epoch <= 50):
                if (epoch % 5 == 0):
                    print("updating gaussian density...")
                    model.update_gaussian_density(training_params, optimizer)
                if (epoch % 15 == 0):
                    print("reseting gaussian aplha...")
                    model.reset_alpha(training_params, optimizer)
            if (epoch % 10 == 0):
                fn = "data/epoch%04d.npy" % epoch
                save_training_params(fn, training_params)
                print("trained data is saved to %s" % fn)

    save_training_params('data/final.npy', training_params)
    print("Training is finished.")
