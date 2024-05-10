from gaussian_splatting import *
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import simple_gaussian_reasterization as sgr
import torchvision

u, cov2d, alpha, \
    depth, color, contrib, final_tau, \
    patch_offset_per_tile, gs_id_per_patch, dloss_dgammas = \
    torch.load("temp.torch")

dloss_dus, dloss_dcov2ds, dloss_dalphas, dloss_dcolors =\
    sgr.backward(546, 979, u, cov2d, alpha,
                 depth, color, contrib, final_tau,
                 patch_offset_per_tile, gs_id_per_patch, dloss_dgammas)
print("u:\n", torch.max(u))
print("cov2d:\n", torch.max(cov2d))
print("alpha:\n", torch.max(alpha))
print("color:\n", torch.max(color))


print("dloss_dus:\n", torch.max(dloss_dus))
print("dloss_dcov2ds:\n", torch.max(dloss_dcov2ds))
print("dloss_dalphas:\n", torch.max(dloss_dalphas))
print("dloss_dcolors:\n", torch.max(dloss_dcolors))
