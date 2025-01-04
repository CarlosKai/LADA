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

        return np.array(cams)  # Shape: (batch_size, sequence_length)

    def mask_important_segments(self, input_tensor, cams):
        """
        Mask time steps with importance higher than the threshold for the entire batch.
        """
        batch_size = input_tensor.size(0)
        masks_low, masks_high = [], []

        for cam in cams:
            cam_tensor = torch.from_numpy(cam).float().unsqueeze(0).unsqueeze(0).to(
                input_tensor.device)  # (1, 1, L_cam)

            # Create masks based on threshold ranges
            mask_low = (cam_tensor <= 0.2).float()  # Retain values in [0, 0.4]
            mask_high = (cam_tensor >= 0.5).float()  # Retain values in [0.6, 1]

            masks_low.append(mask_low)  # Append mask for low importance
            masks_high.append(mask_high)  # Append mask for high importance

        # Concatenate masks for the batch
        masks_low = torch.cat(masks_low, dim=0)  # Shape: (batch_size, 1, sequence_length)
        masks_high = torch.cat(masks_high, dim=0)  # Shape: (batch_size, 1, sequence_length)

        # Apply masks to input tensor
        masked_input_low = input_tensor * masks_low  # Retain low-importance segments
        masked_input_high = input_tensor * masks_high  # Retain high-importance segments

        return masked_input_low, masked_input_high  # Shape: (batch_size, num_channels, sequence_length)