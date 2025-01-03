import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt


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
            # nn.MaxPool1d(kernel_size=2)  # Output length: 128 -> 64
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # Output length: 64 -> 32
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = torch.mean(x, dim=2)  # Global Average Pooling
        x = self.fc(x)
        return x


# Grad-CAM Implementation
# class GradCAM:
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#         self.gradients = None
#         self.activations = None
#         self._register_hooks()
#
#     def _register_hooks(self):
#         def forward_hook(module, input, output):
#             self.activations = output  # Save the feature maps
#
#         def backward_hook(module, grad_in, grad_out):
#             self.gradients = grad_out[0]  # Save the gradients
#
#         self.target_layer.register_forward_hook(forward_hook)
#         self.target_layer.register_backward_hook(backward_hook)
#
#     def generate_cam(self, input_tensor, target_class=None):
#         input_tensor.requires_grad = True
#         outputs = self.model(input_tensor)
#         if target_class is None:
#             target_class = outputs.argmax(dim=1)  # Predicted class
#
#         scores = outputs[:, target_class]
#         scores.backward(torch.ones_like(scores))
#
#         weights = self.gradients.mean(dim=2, keepdim=True)  # Shape: (batch_size, num_channels, 1)
#         cam = weights * self.activations  # Shape: (batch_size, num_channels, sequence_length)
#         cam = F.relu(cam).mean(dim=1).squeeze().detach().cpu().numpy()  # Average over channels
#
#         # Normalize CAM
#         cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
#
#         return cam  # Shape: (sequence_length,)
#
#     def mask_important_segments(self, input_tensor, cam, threshold=0.5):
#         cam = torch.from_numpy(cam).float().unsqueeze(0).unsqueeze(0).to(input_tensor.device)  # Shape: (1, 1, L_cam)
#         cam_resized = F.interpolate(cam, size=input_tensor.size(2), mode='linear', align_corners=False).squeeze()
#         mask = (cam_resized < threshold).float()  # 低于阈值的时间步被保留
#         mask = mask.unsqueeze(0).unsqueeze(0)
#         masked_input = input_tensor * mask  # Apply the mask to the input
#         return masked_input
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
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
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        input_tensor.requires_grad = True
        outputs = self.model(input_tensor)  # Shape: (batch_size, num_classes)

        # If target class is not provided, use the predicted class for each sample
        if target_class is None:
            target_class = outputs.argmax(dim=1)  # Shape: (batch_size,)

        # Create a tensor of scores for each sample in the batch
        scores = torch.gather(outputs, 1, target_class.view(-1, 1)).squeeze()  # Shape: (batch_size,)
        scores.backward(torch.ones_like(scores))  # Compute gradients

        # Compute weights by averaging gradients over the time dimension
        weights = self.gradients.mean(dim=2, keepdim=True)  # Shape: (batch_size, num_channels, 1)
        cam = weights * self.activations  # Shape: (batch_size, num_channels, sequence_length)
        cam = F.relu(cam).mean(
            dim=1).detach().cpu().numpy()  # Average over channels, shape: (batch_size, sequence_length)

        # Normalize CAM for each sample in the batch
        cams = []
        for c in cam:
            c = (c - c.min()) / (c.max() - c.min() + 1e-8)
            cams.append(c)

        return np.array(cams)  # Shape: (batch_size, sequence_length)

    def mask_important_segments(self, input_tensor, cams, threshold=0.5):
        """
        Mask time steps with importance higher than the threshold for the entire batch.
        """
        batch_size = input_tensor.size(0)
        masks = []
        for cam in cams:
            cam_tensor = torch.from_numpy(cam).float().unsqueeze(0).unsqueeze(0).to(
                input_tensor.device)  # (1, 1, L_cam)
            cam_resized = F.interpolate(cam_tensor, size=input_tensor.size(2), mode='linear',
                                        align_corners=False).squeeze()  # (sequence_length,)
            mask = (cam_resized < threshold).float().unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, sequence_length)
            masks.append(mask)
        masks = torch.cat(masks, dim=0)  # Shape: (batch_size, 1, sequence_length)
        masked_input = input_tensor * masks  # Apply mask for each sample in the batch
        return masked_input  # Shape: (batch_size, num_channels, sequence_length)


# Visualization function
def visualize_cam(input_seq, cam, title="Grad-CAM Importance Over Time Steps"):
    plt.figure(figsize=(10, 4))
    plt.plot(input_seq[0, 0].cpu().detach().numpy(), label='Original Input')
    plt.plot(cam, label='Grad-CAM Importance', color='red', alpha=0.6)
    plt.title(title)
    plt.legend()
    plt.show()


# Main Function
def main():
    # Simulated data
    num_samples = 1000
    num_classes = 5
    input_channels = 144  # Number of features
    input_length = 128  # Sequence length

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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model, loss, and optimizer
    model = TCN(input_channels=input_channels, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    for epoch in range(5):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    print("Training complete!")

    # Grad-CAM analysis
    model.eval()
    test_sample = torch.tensor(test_data[0:1], dtype=torch.float32).to(device)  # First test sample
    target_layer = model.conv_block2[0]  # Second convolution layer
    grad_cam = GradCAM(model, target_layer)

    # Generate CAM and mask important segments
    test_samples = torch.tensor(test_data[0:8], dtype=torch.float32).to(device)  # First 8 test samples
    time_step_importances = grad_cam.generate_cam(test_samples)
    masked_inputs = grad_cam.mask_important_segments(test_samples, time_step_importances, threshold=0.5)

    # Pass masked inputs through the model
    masked_outputs = model(masked_inputs)
    print(f"Original Output: {model(test_samples)}")
    print(f"Masked Output: {masked_outputs}")

if __name__ == "__main__":
    main()
