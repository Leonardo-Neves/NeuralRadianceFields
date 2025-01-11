import torch

def compute_accumulated_transmittance(betas):
    accumulated_transmittance = torch.cumprod(betas, 1)     
    return torch.cat((torch.ones(accumulated_transmittance.shape[0], 1, device=accumulated_transmittance.device),
                      accumulated_transmittance[:, :-1]), dim=1)

def rendering(model, rays_o, rays_d, tn, tf, nb_bins=100, device='cpu', white_bckgr=True):
    
    t = torch.linspace(tn, tf, nb_bins).to(device) # [nb_bins]

    # delta -> t_{i+1} - t_{i} = On the code (t_{i+1} - t_{-1}) + torch.tensor([1e10](very large number))
    delta = torch.cat((t[1:] - t[:-1], torch.tensor([1e10], device=device)))
    
    # r(t) = o + t * d
    x = rays_o.unsqueeze(1) + t.unsqueeze(0).unsqueeze(-1) * rays_d.unsqueeze(1) # [nb_rays, nb_bins, 3]    
    
    # Results of using the sigma(density) and the color function
    # Input model.intersect -> xyz = [nb_rays, 3]
    colors, density = model.intersect(x.reshape(-1, 3), rays_d.expand(x.shape[1], x.shape[0], 3).transpose(0, 1).reshape(-1, 3))
    
    colors = colors.reshape((x.shape[0], nb_bins, 3)) # [nb_rays, nb_bins, 3] [H*W, 100, 3]
    density = density.reshape((x.shape[0], nb_bins)) # [H*W, 100]
    
    # T(t) = exp(- \int_{0}^{t} \sigma(r(t)) dt)
    alpha = 1 - torch.exp(- density * delta.unsqueeze(0)) # [nb_rays, nb_bins, 1]
        
    weights = compute_accumulated_transmittance(1 - alpha) * alpha # [nb_rays, nb_bins] for each ray we have 100 samples
    
    if white_bckgr:
        c = (weights.unsqueeze(-1) * colors).sum(1) # [nb_rays, 3]
        weight_sum = weights.sum(-1) # [nb_rays]
        return c + 1 - weight_sum.unsqueeze(-1)
    else:
        c = (weights.unsqueeze(-1) * colors).sum(1) # [nb_rays, 3]
    
    return c