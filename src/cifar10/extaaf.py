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


# Define Extended TAAF Function (ExTAAF)
class ExTAAF(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.beta = nn.Parameter(torch.tensor(float(beta)))

    def forward(self, x):
        exponential_term = x * torch.exp(-self.alpha * (x**2))
        quadratic_term = self.beta * (x**2) / (1 + torch.abs(x))
        return exponential_term + quadratic_term


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


# Define DISEM10 Activation Function
class DISEM10(nn.Module):
    def forward(self, x):
        return torch.clamp(
            torch.tanh(
                torch.sinh(x) * torch.cos(x)
                + torch.tanh(x) * torch.exp(torch.tensor(-1.0))
            )
            * 2,
            -3,
            3,
        )


# Define a simple CNN model for CIFAR-10 with different activation options
class CIFAR10Model(nn.Module):
    def __init__(
        self,
        activation_type="ELU",
        elu_alpha=1.0,
        elu_beta=1.0,
        extaaf_alpha=1.0,
        extaaf_beta=1.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.activation_type = activation_type

        if activation_type == "TAAF":
            self.activation = TAAF()
        elif activation_type == "ExTAAF":
            self.activation = ExTAAF(alpha=extaaf_alpha, beta=extaaf_beta)
        elif activation_type == "ELU":
            self.activation = ELU(alpha=elu_alpha, beta=elu_beta)
        elif activation_type == "DISEM10":
            self.activation = DISEM10()
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
        else:  # ELU or TAAF or ExTAAF are used directly
            x = self.pool(self.activation(self.conv1(x)))  # Conv1 -> Activation -> Pool
            x = self.pool(self.activation(self.conv2(x)))  # Conv2 -> Activation -> Pool
            x = x.view(x.size(0), -1)  # Flatten
            x = self.activation(self.fc1(x))  # FC1 -> Activation
        x = self.fc2(x)  # FC2 (no activation for output layer)
        return x


# Function to calculate Emergent Linearity Coefficient (ELC)
def calculate_elc(model, data_loader, delta, device):
    model.eval()  # Set model to evaluation mode
    total_elc = 0.0
    num_samples = 0
    with torch.no_grad():
        for inputs, _ in data_loader:  # No labels needed for ELC calculation
            inputs = inputs.to(device)
            for i in range(inputs.size(0)):  # Iterate over batch samples
                x_i = inputs[i].unsqueeze(
                    0
                )  # Get single sample and add batch dimension
                f_x_i = model(x_i).sum()  # Get model output for x_i, sum to get scalar

                x_i_perturbed = x_i + delta * torch.randn_like(x_i)  # Perturb input
                x_i_perturbed = x_i_perturbed.to(device)
                f_x_i_delta = model(
                    x_i_perturbed
                ).sum()  # Get model output for perturbed x_i

                elc_sample = torch.abs(f_x_i_delta - f_x_i) / delta
                total_elc += elc_sample.item()
                num_samples += 1
    return total_elc / num_samples if num_samples > 0 else 0.0


# Training loop (modified to include ELC calculation)
def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
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

    # Calculate ELC every epoch, using a small subset of the training data for efficiency
    elc_delta = 0.01  # Adjust delta as needed
    elc_train_loader_subset = DataLoader(
        torch.utils.data.Subset(
            train_dataset, range(0, len(train_dataset), 100)
        ),  # Use every 100th sample for ELC
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )  # Reduced batch size and num_workers for ELC calculation
    elc_value = calculate_elc(model, elc_train_loader_subset, elc_delta, device)

    return (
        train_loss,
        top1_train_accuracy,
        percentage_train_error,
        elc_value,
    )  # Return elc_value


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


if __name__ == "__main__":  # ADD THIS BLOCK
    # Initialize device - moved to be inside __main__
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")  # Print device info

    if str(device) == "cuda":
        assert torch.cuda.is_available(), (
            "CUDA is not available but 'cuda' device was specified. Please check your CUDA installation and GPU."
        )
    else:
        print(
            "WARNING: Training on CPU. This will be significantly slower. Consider using a GPU if available."
        )

    # Load CIFAR-10 dataset (same as before)
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

    # --- Choose Activation Type Here ---
    activation_type = "ExTAAF"  # Options: 'TAAF', 'ExTAAF', 'ELU', 'DISEM10', 'Tanh', 'ReLU', 'Sigmoid', 'GELU', 'Combined'
    elu_alpha = 1.0  # Adjust alpha for ELU
    elu_beta = 1.0  # Adjust beta for ELU
    extaaf_alpha = 1.0  # Adjust alpha for ExTAAF
    extaaf_beta = 1.0  # Adjust beta for ExTAAF

    # --- End Activation Type Choice ---

    model = CIFAR10Model(
        activation_type=activation_type,
        elu_alpha=elu_alpha,
        elu_beta=elu_beta,
        extaaf_alpha=extaaf_alpha,
        extaaf_beta=extaaf_beta,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)  # Updated Optimizer to AdamW

    # Learning Rate Scheduler - Exponential Decay (same as before)
    lr_scheduler = ExponentialLR(optimizer, gamma=0.98)  # Decay rate of 0.98 per epoch

    # Train and evaluate the model (modified to print ELC)
    num_epochs = 300  # Increased epochs for CIFAR-10
    model_name_suffix = activation_type
    model_dir = "./tests/cifar10/"
    if activation_type == "ELU":
        model_name_suffix += f"_alpha{elu_alpha}_beta{elu_beta}"
    elif activation_type == "ExTAAF":
        model_name_suffix += f"_alpha{extaaf_alpha}_beta{extaaf_beta}"

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs} - Activation: {activation_type}")
        (
            train_loss,
            top1_train_accuracy,
            percentage_train_error,
            elc_value,
        ) = train(  # Get elc_value from train function
            model, train_loader, criterion, optimizer, device, epoch
        )
        test_loss, top1_accuracy, percentage_error = test(
            model, test_loader, criterion, device
        )
        print(
            f"Train Loss: {train_loss:.4f}, Train Top-1 Accuracy: {top1_train_accuracy:.2f}%, Train Percentage Error: {percentage_train_error:.2f}%"
        )
        print(
            f"Test Loss: {test_loss:.4f}, Test Top-1 Accuracy: {top1_accuracy:.2f}%, Test Percentage Error: {percentage_error:.2f}%"
        )
        print(
            f"Emergent Linearity Coefficient (ELC): {elc_value:.4f}"
        )  # Print ELC value

        lr_scheduler.step()  # Step the learning rate scheduler every epoch

    # Save the model checkpoint (filename now includes activation type)
    model_filename = f"{model_dir}cifar10_{model_name_suffix}_model_300epochs_AdamW.pth"  # Updated filename
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved to {model_filename}")
