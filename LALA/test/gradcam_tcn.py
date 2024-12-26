import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
from matplotlib import pyplot as plt


# Dataset for Time-Series Data
class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data  # Shape: (N, C, L)
        self.labels = labels  # Shape: (N,)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


# TCN Model Definition
class TCN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(TCN, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = torch.mean(x, dim=2)  # Global Average Pooling
        x = self.fc(x)
        return x


# Grad-CAM Class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        input_tensor.requires_grad = True
        outputs = self.model(input_tensor)  # Forward pass
        if target_class is None:
            target_class = outputs.argmax(dim=1)  # Predicted class

        scores = outputs[:, target_class]  # Class scores
        scores.backward(torch.ones_like(scores))  # Backward pass

        weights = self.gradients.mean(dim=2, keepdim=True)  # Global average pooling每个通道在全局上的权重
        # cam = (weights * self.activations).sum(dim=1)  # Weighted sum of activations
        # cam = F.relu(cam)  # ReLU to keep only positive values
        #
        # # Normalize and interpolate to match input size
        # cam = cam.unsqueeze(1)  # Add channel dimension
        # cam = F.interpolate(cam, size=(input_tensor.size(2)), mode='linear', align_corners=False)
        # cam = cam.squeeze()  # Remove channel dimension
        # cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # Normalize between 0 and 1
        # return cam

        # Compute channel-wise Grad-CAM by multiplying weights with activations (no sum over channels)
        cam = weights * self.activations  # Shape: (batch_size, num_channels, sequence_length)

        # Apply ReLU to keep only positive contributions
        cam = F.relu(cam)
        # Normalize each channel's heatmap individually
        cam = cam.cpu().detach().numpy()  # Convert to NumPy for further processing
        batch_size, num_channels, sequence_length = cam.shape
        cam_normalized = np.zeros_like(cam)  # To store normalized heatmaps

        for i in range(num_channels):  # Normalize each channel separately
            channel_heatmap = cam[:, i, :]
            channel_heatmap = (channel_heatmap - channel_heatmap.min()) / (
                        channel_heatmap.max() - channel_heatmap.min() + 1e-8)
            cam_normalized[:, i, :] = channel_heatmap

        return cam_normalized  # Shape: (batch_size, num_channels, sequence_length)


# Training Function
def train_model(model, train_loader, criterion, optimizer, epochs=10, device='cuda'):
    model = model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")
    print("Training complete!")


# Generate Grad-CAM Heatmap
def generate_gradcam_heatmap(model, target_layer, input_tensor, device='cuda'):
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam.generate_cam(input_tensor)

    # Convert Grad-CAM to heatmaps for each channel
    cam = cam[0]  # Select the first sample in the batch (shape: [num_channels, sequence_length])
    heatmaps = []

    for channel in range(cam.shape[0]):
        # Convert to heatmap (0-255 range)
        heatmap = np.uint8(255 * cam[channel])
        heatmaps.append(heatmap)

    # Combine all channel heatmaps into a single 9x128 image (optional)
    combined_heatmap = np.stack(heatmaps, axis=0)  # Shape: (num_channels, sequence_length)

    return combined_heatmap  # Shape: (9, 128)

    # Convert Grad-CAM to heatmap
    # cam = cam.cpu().detach().numpy()  # Convert to NumPy
    # heatmap = np.uint8(255 * cam)
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    #
    # # Overlay heatmap on input
    # input_tensor_np = input_tensor.cpu().detach().numpy().squeeze()  # Shape: (9, 128)
    # input_tensor_np = input_tensor_np.mean(axis=0)  # Average over channels for visualization
    # input_tensor_np = np.uint8(
    #     255 * (input_tensor_np - input_tensor_np.min()) / (input_tensor_np.max() - input_tensor_np.min()))
    #
    # # Convert input_tensor_np to 3 channels to match heatmap
    # input_tensor_np = cv2.cvtColor(input_tensor_np, cv2.COLOR_GRAY2BGR)
    #
    # # Resize input_tensor_np to match heatmap
    # input_tensor_np = cv2.resize(input_tensor_np, (heatmap.shape[1], heatmap.shape[0]))
    #
    # overlay = cv2.addWeighted(input_tensor_np, 0.6, heatmap, 0.4, 0)
    # return heatmap, overlay


# Main Script
if __name__ == "__main__":
    # Simulated data
    num_samples = 1000
    num_classes = 5
    input_channels = 9
    input_length = 128

    # Generate random time-series data
    data = np.random.rand(num_samples, input_channels, input_length).astype(np.float32)
    labels = np.random.randint(0, num_classes, size=num_samples)

    # Train-Test Split
    train_data = data[:800]
    train_labels = labels[:800]
    test_data = data[800:]
    test_labels = labels[800:]

    # Create datasets and dataloaders
    train_dataset = TimeSeriesDataset(train_data, train_labels)
    test_dataset = TimeSeriesDataset(test_data, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model, loss, and optimizer
    model = TCN(input_channels=input_channels, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, epochs=10)

    # Generate Grad-CAM for a test sample
    test_sample = torch.tensor(test_data[0:1], dtype=torch.float32)  # Use the first test sample
    target_layer = model.conv_block2[0]  # Choose the target layer
    heatmap, overlay = generate_gradcam_heatmap(model, target_layer, test_sample)

    # Save and display Grad-CAM results
    cv2.imwrite('gradcam_heatmap.png', heatmap)
    cv2.imwrite('gradcam_overlay.png', overlay)

    plt.imshow(overlay, cmap='jet')
    plt.title("Grad-CAM Heatmap Overlay")
    plt.show()
