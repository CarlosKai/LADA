import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

class GradTCN:
    def __init__(self, model1, model2, target_layer):
        self.model = nn.Sequential(model1, model2)
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output  # Save the feature maps for the batch

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]  # Save the gradients for the batch

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class):
        # input_tensor.requires_grad = True
        outputs = self.model(input_tensor)  # Shape: (batch_size, num_classes)

        # # If target class is not provided, use the predicted class for each sample
        # if target_class is None:
        #     target_class = outputs.argmax(dim=1)  # Shape: (batch_size,)

        # Create a tensor of scores for each sample in the batch
        scores = torch.gather(outputs, 1, target_class.view(-1, 1)).squeeze()  # Shape: (batch_size,)
        scores.backward(gradient=torch.ones_like(scores), retain_graph=True)  # Compute gradients

        # Compute weights by averaging gradients over the time dimension
        weights = self.gradients.mean(dim=2, keepdim=True)  # Shape: (batch_size, num_channels, 1)
        cam = weights * self.activations  # Shape: (batch_size, num_channels, sequence_length)
        cam = F.relu(cam).mean(
            dim=1).detach().cpu().numpy()  # Average over channels, shape: (batch_size, sequence_length)
        cam = cam[:, :128]
        # Normalize CAM for each sample in the batch
        cams = []
        for c in cam:
            c = (c - c.min()) / (c.max() - c.min() + 1e-8)
            cams.append(c)

        self.gradients = None
        self.activations = None

        return np.array(cams)  # Shape: (batch_size, sequence_length)

    def mask_important_segments(self, input_tensor, cams):

        batch_size, num_channels, sequence_length = input_tensor.size()

        # Initialize masks for the entire batch
        masks_low = torch.zeros(batch_size, 1, sequence_length, device=input_tensor.device)
        masks_high = torch.zeros(batch_size, 1, sequence_length, device=input_tensor.device)

        # Create masks for each sample in the batch
        for i, cam in enumerate(cams):
            cam_tensor = torch.from_numpy(cam).float().unsqueeze(0).unsqueeze(0).to(
                input_tensor.device)  # Shape: (1, 1, L_cam)

            # Calculate the 20th and 80th percentiles for the current CAM
            low_threshold = torch.quantile(cam_tensor, 0.2)
            high_threshold = torch.quantile(cam_tensor, 0.8)

            # Create masks for the lowest 20% and highest 20%
            masks_low[i] = (cam_tensor <= low_threshold).float()  # Mask for lowest 20%
            masks_high[i] = (cam_tensor >= high_threshold).float()  # Mask for highest 20%

        # Scale the input tensor
        masked_input_low = input_tensor * (1 - masks_low * 0.8)  # Scale lowest 20% to 0.2
        masked_input_high = input_tensor * (1 - masks_high * 0.8)  # Scale highest 20% to 0.2

        return masked_input_low, masked_input_high  # Shape: (batch_size, num_channels, sequence_length)
