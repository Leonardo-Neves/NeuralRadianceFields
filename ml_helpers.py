from tqdm import tqdm
from rendering import rendering
import torch

def training(model, optimizer, scheduler, tn, tf, nb_bins, nb_epochs, data_loader, device='cpu'):
    
    training_loss = []
    for epoch in (range(nb_epochs)):
        for batch in tqdm(data_loader):
            o = batch[:, :3].to(device)
            d = batch[:, 3:6].to(device)
            
            target = batch[:, 6:].to(device)
            
            # Initialy the model uses random values for the voxels
            # The model is updated by the optimizer
            # The initial o and d are from real data
            prediction = rendering(model, o, d, tn, tf, nb_bins=nb_bins, device=device)
            
            loss = ((prediction - target)**2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())

            '''
                optimizer.zero_grad()

                - Before computing a new gradient, we zero out (reset) all the gradients in the model's parameters.
                - If we don't zero them, PyTorch would accumulate gradients from each .backward() call.

                loss.backward()

                - Computes the derivative of the loss with respect to each parameter in the model that has requires_grad=True.
                - This is the actual “backpropagation” step.
                - After this call, each parameter's .grad field holds the computed gradient.
                
                optimizer.step()

                - Uses the gradients computed from .backward() to update the model's parameters.
                - The exact update rule depends on the optimizer (e.g., SGD, Adam).
                - This is where the learning actually happens.

                training_loss.append(loss.item())

                - loss.item() converts the loss tensor (containing a single scalar value) into a regular Python float.
                - Appending that value to a list (here, training_loss) is a way to log or keep track of the loss as 
                  training progresses, for later visualization or analysis.
            '''
            
        scheduler.step()
        
        torch.save(model.cpu(), 'model_nerf')
        model.to(device)
        
    return training_loss