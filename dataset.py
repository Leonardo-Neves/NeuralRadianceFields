import torch
import numpy as np
import os
import imageio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def get_rays(datapath, mode='train'):
    
    pose_file_names = [f for f in os.listdir(datapath + f'/{mode}/pose') if f.endswith('.txt')]
    intrisics_file_names = [f for f in os.listdir(datapath + f'/{mode}/intrinsics') if f.endswith('.txt')]
    img_file_names = [f for f in os.listdir(datapath + '/imgs') if mode in f]

    assert len(pose_file_names) == len(intrisics_file_names)
    assert len(img_file_names) == len(pose_file_names)
    
    # Read
    N = len(pose_file_names)
    poses = np.zeros((N, 4, 4))
    intrinsics = np.zeros((N, 4, 4))
    
    images = []
    
    for i in range(N):
        name = pose_file_names[i]
        
        pose = open(datapath + f'/{mode}/pose/' + name).read().split()
        poses[i] = np.array(pose, dtype=float).reshape(4, 4)
        
        intrinsic = open(datapath + f'/{mode}/intrinsics/' + name).read().split()
        intrinsics[i] = np.array(intrinsic, dtype=float).reshape(4, 4)
        
        # Read images
        img = imageio.imread(datapath + '/imgs/' + name.replace('txt', 'png')) / 255.
        images.append(img[None, ...])

    images = np.concatenate(images)
    
    H = images.shape[1]
    W = images.shape[2]
    
    if images.shape[3] == 4: #RGBA -> RGB
        images = images[..., :3] * images[..., -1:] + (1 - images[..., -1:])

    rays_o = np.zeros((N, H*W, 3))
    rays_d = np.zeros((N, H*W, 3))
    target_px_values = images.reshape((N, H*W, 3))
    
    for i in range(N):
        
        c2w = poses[i]
        f = intrinsics[i, 0, 0]

        u = np.arange(W)
        v = np.arange(H)
        u, v = np.meshgrid(u, v)
        dirs = np.stack((
            u - W / 2,             # x
            -(v - H / 2),          # y
            - np.ones_like(u) * f  # z
        ), axis=-1)

        '''
            @ is a standard matrix multiplication like np.dot

            [..., None] (or equivalently [..., np.newaxis]) adds an extra dimension along the last axis of the array.
            (c2w[:3, :3] @ dirs[..., None]) is [400, 400, 3, 1], squeeze(-1) removes the last dimension

            All other “leading” dimensions (e.g. (H, W)) are treated as batch dimensions and are broadcast accordingly.

            When we talk about matrix multiplication with multidimensional arrays, NumPy will treat all dimensions before
            the last two as “batch” (or “batching”) dimensions. It effectively performs the matrix multiplication on each 
            slice of the array along those leading dimensions. For example, a shape (H, W, 3, 3) multiplied by a shape 
            (H, W, 3, 1) means NumPy does (3×3) @ (3×1) for each of the (H, W) positions, giving a (H, W, 3, 1) result.
        '''
        dirs = (c2w[:3, :3] @ dirs[..., None]).squeeze(-1)

        # np.linalg.norm calculate the euclidian distance or any other measure of distance
        # -1 means the last axis
        dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

        rays_d[i] = dirs.reshape(-1, 3) # [400, 400, 3] -> [160000, 3]
        rays_o[i] += c2w[:3, 3] # c2w[:3, 3] is a rotation matrix

    return rays_o, rays_d, target_px_values
