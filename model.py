import torch
import torch.nn as nn

class Voxels(nn.Module):
    
    def __init__(self, nb_voxels=100, scale=1, device='cpu'):
        super(Voxels, self).__init__()
        # 4 [R, G, B, density]
        # nb_voxels = nb_bins
        # nb_voxels is the number of samples in each dimensions xyz, because it is a cube
        self.voxels = torch.nn.Parameter(torch.rand(
            (nb_voxels, nb_voxels, nb_voxels, 4), 
            device=device, 
            requires_grad=True
        ))

        '''
            In PyTorch, any tensor you want the optimizer to update (i.e., learnable weights) needs 
            to be registered as a Parameter in an nn.Module. By wrapping your voxel tensor 
            with torch.nn.Parameter(...), you ensure:

            - It shows up in model.parameters() – so that any optimizer (like Adam or SGD) can see and update it.
            - It participates in autograd – PyTorch will track gradients for it and backpropagate changes through it.

            If you just stored self.voxels as a regular torch.Tensor, it would not be recognized as part of the model’s 
            trainable parameters, and thus it would remain fixed during training.
        '''
        
        self.nb_voxels = nb_voxels
        self.device = device
        self.scale = scale
        '''
            The scale is the size of the cube, so if the scale is 1, the 
            cube will have a size of 1x1x1
            The scale is used to normalize the values of the voxels, 
            so the values of the voxels are between -0.5 and 0.5

            E.g.: If the scale is 1, betweeh -0.5 and 0.5 we have 100 voxels
        '''
        
    def forward(self, xyz, d):
        
        x = xyz[:, 0] # [nb_rays, 3]
        y = xyz[:, 1] # [nb_rays, 3]
        z = xyz[:, 2] # [nb_rays, 3]
        
        '''
            The scale have to be divided by 2 because the values of the voxels are between -0.5 and 0.5
        '''
        cond = (x.abs() < (self.scale / 2)) & (y.abs() < (self.scale / 2)) & (z.abs() < (self.scale / 2))
        
        indx = (x[cond] / (self.scale / self.nb_voxels) + self.nb_voxels / 2).type(torch.long)
        indy = (y[cond] / (self.scale / self.nb_voxels) + self.nb_voxels / 2).type(torch.long)
        indz = (z[cond] / (self.scale / self.nb_voxels) + self.nb_voxels / 2).type(torch.long)
        
        # (xyz.shape[0], 4) -> [R, G, B, density]
        colors_and_densities = torch.zeros((xyz.shape[0], 4), device=xyz.device)
        colors_and_densities[cond, :3] = self.voxels[indx, indy, indz, :3] # RGB
        colors_and_densities[cond, -1] = self.voxels[indx, indy, indz, -1] # Density
        
        # return sigma/density and colors
        # sigmoid output dimension -> [0, 1] and relu output dimension -> [0, inf]
        # color must be between 0 and 1 and the density must be between 0 and inf
        return torch.sigmoid(colors_and_densities[:, :3]), torch.relu(colors_and_densities[:, -1:])
        
    
    def intersect(self, x, d):
        return self.forward(x, d)
