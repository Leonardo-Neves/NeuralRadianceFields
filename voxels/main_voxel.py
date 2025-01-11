import torch
import torch.nn as nn
import numpy as np

import os
import imageio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import get_rays
from rendering import rendering
from model import Voxels
from ml_helpers import training


batch_size = 1024

o, d, target_px_values = get_rays(r'C:\Users\leosn\Desktop\NERF\fox', mode='train')

dataloader = DataLoader(
    torch.cat((torch.from_numpy(o).reshape(-1, 3),
    torch.from_numpy(d).reshape(-1, 3),
    torch.from_numpy(target_px_values).reshape(-1, 3)), dim=1),
    batch_size=batch_size, 
    shuffle=True
)

model = Voxels(nb_voxels=100, scale=1, device='cpu')

img = rendering(model, torch.from_numpy(o[2]), torch.from_numpy(d[2]), 8, 12)

plt.imshow(img.reshape(400, 400, 3).data.cpu().numpy())
plt.show()
