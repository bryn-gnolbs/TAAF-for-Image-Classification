# --------------------------------------------------------------------------------------------------
# This code is released under the terms of the Hippocratic License 3.0.
#
# Hippocratic License 3.0: Modified for Bryn T. Chatfield & TAAF Project
# Version 3.0, Modified February 14, 2025
# Full license text available in the LICENSE file in this repository and at:
# https://opensourcemodels.com/licenses/hippocratic-license-3.0-modified-taaf.txt
#
# Copyright (c) 2025, Bryn T. Chatfield
# --------------------------------------------------------------------------------------------------

import torch  # noqa: N999
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.nn.functional as F  # Import for functional tanh
from torch.optim.lr_scheduler import ExponentialLR  # Import for LR Decay


# Define The Analog Activation Function (TAAF)
class TAAF(nn.Module):
    def forward(self, x):
        numerator = torch.exp(-x)
        denominator = torch.exp(-x) + torch.exp(-(x**2))  # Sum of e^{-x} and e^{-x^2}
        return (numerator / denominator) - (1 / 2)  # TAAF formula


# Define the Emergent Linearizing Unit (ELU) activation function
class ELU(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = nn.Parameter(
            torch.tensor(float(alpha))
        )  # Make alpha learnable if needed
        self.beta = nn.Parameter(
            torch.tensor(float(beta))
        )  # Make beta learnable if needed

    def forward(self, x):
        return torch.tanh(self.alpha * x) + self.beta * (x / (1 + torch.abs(x)))


# Define a simple CNN model for CIFAR-10 with ELU (or TAAF/Combined)
class CIFAR10Model(nn.Module):
    def __init__(self, activation_type="ELU", elu_alpha=1.0, elu_beta=1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(
            3, 32, kernel_size=3, stride=1, padding=1
        )  # Input channels changed to 3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(
            64 * 8 * 8, 128
        )  # Input size adjusted for CIFAR-10 (8x8 feature map)
        self.fc2 = nn.Linear(128, 10)
        self.activation_type = activation_type

        if activation_type == "TAAF":
            self.activation = TAAF()
        elif activation_type == "ELU":
            self.activation = ELU(alpha=elu_alpha, beta=elu_beta)
        elif activation_type == "Tanh":
            self.activation = nn.Tanh()
        elif activation_type == "ReLU":
            self.activation = nn.ReLU()
        elif activation_type == "Sigmoid":
            self.activation = nn.Sigmoid()
        elif activation_type == "GELU":
            self.activation = nn.GELU()
        elif (
            activation_type == "Combined"
        ):  # Example Combined Activation (ELU then TAAF)
            self.elu_activation = ELU(alpha=elu_alpha, beta=elu_beta)
            self.taaf_activation = TAAF()
            self.activation = lambda x: self.taaf_activation(
                self.elu_activation(x)
            )  # Define combined in forward
        else:
            raise ValueError(f"Activation type '{activation_type}' not recognized.")

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if (
            self.activation_type == "Combined"
        ):  # Combined activation is applied sequentially
            x = self.pool(
                self.activation(self.conv1(x))
            )  # Conv1 -> ELU -> TAAF -> Pool
            x = self.pool(
                self.activation(self.conv2(x))
            )  # Conv2 -> ELU -> TAAF -> Pool
            x = x.view(x.size(0), -1)  # Flatten
            x = self.activation(self.fc1(x))  # FC1 -> ELU -> TAAF
        else:  # ELU or TAAF are used directly
            x = self.pool(self.activation(self.conv1(x)))  # Conv1 -> Activation -> Pool
            x = self.pool(self.activation(self.conv2(x)))  # Conv2 -> Activation -> Pool
            x = x.view(x.size(0), -1)  # Flatten
            x = self.activation(self.fc1(x))  # FC1 -> Activation
        x = self.fc2(x)  # FC2 (no activation for output layer)
        return x


# Load CIFAR-10 dataset
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
        ),  # CIFAR-10 normalization
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
        ),  # CIFAR-10 normalization
    ]
)


train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)
test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)

train_loader = DataLoader(
    train_dataset, batch_size=120, shuffle=True, num_workers=2
)  # Updated batch_size to 120
test_loader = DataLoader(
    test_dataset, batch_size=120, shuffle=False, num_workers=2
)  # Updated batch_size to 120

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Choose Activation Type Here ---
activation_type = (
    "TAAF"  # Options: 'TAAF', 'ELU', 'Combined', 'Tanh', 'ReLU', 'Sigmoid', 'GELU'
)
elu_alpha = 1.0  # Adjust alpha for ELU
elu_beta = 1.0  # Adjust beta for ELU
# --- End Activation Type Choice ---

model = CIFAR10Model(
    activation_type=activation_type, elu_alpha=elu_alpha, elu_beta=elu_beta
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)  # Updated Optimizer to AdamW

# Learning Rate Scheduler - Exponential Decay
lr_scheduler = ExponentialLR(optimizer, gamma=0.98)  # Decay rate of 0.98 per epoch


# Training loop (same as before)
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)  # noqa: PLW2901
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    top1_train_accuracy = 100.0 * correct / total  # Renamed to top1_train_accuracy
    percentage_train_error = 100.0 - top1_train_accuracy  # Calculate Percentage error
    return (
        train_loss,
        top1_train_accuracy,
        percentage_train_error,
    )  # Return percentage_train_error


# Testing loop (same as before)
def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)  # noqa: PLW2901
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = running_loss / len(test_loader)
    top1_accuracy = 100.0 * correct / total  # Renamed to top1_accuracy
    percentage_error = 100.0 - top1_accuracy  # Calculate Percentage error
    return test_loss, top1_accuracy, percentage_error  # Return percentage_error


# Train and evaluate the model (same as before with model name change)
num_epochs = 300  # Increased epochs for CIFAR-10
model_name_suffix = activation_type
model_dir = "./tests/cifar10/"
if activation_type == "ELU":
    model_name_suffix += f"_alpha{elu_alpha}_beta{elu_beta}"

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs} - Activation: {activation_type}")
    train_loss, top1_train_accuracy, percentage_train_error = train(
        model, train_loader, criterion, optimizer, device
    )  # Get percentage_train_error
    test_loss, top1_accuracy, percentage_error = test(
        model, test_loader, criterion, device
    )  # Get percentage_error
    print(
        f"Train Loss: {train_loss:.4f}, Train Top-1 Accuracy: {top1_train_accuracy:.2f}%, Train Percentage Error: {percentage_train_error:.2f}%"
    )  # Updated print statement
    print(
        f"Test Loss: {test_loss:.4f}, Test Top-1 Accuracy: {top1_accuracy:.2f}%, Test Percentage Error: {percentage_error:.2f}%"
    )  # Updated print statement

    lr_scheduler.step()  # Step the learning rate scheduler every epoch

# Save the model checkpoint (filename now includes activation type)
model_filename = f"{model_dir}cifar10_{model_name_suffix}_model_300epochs_AdamW.pth"  # Updated filename
torch.save(model.state_dict(), model_filename)
print(f"Model saved to {model_filename}")
