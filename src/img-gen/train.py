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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import (
    datasets,
)  # Import torchvision datasets instead of huggingface datasets
import torchvision.transforms as transforms  # For image resizing and transforms
from torchvision.utils import save_image  # For saving images
import os  # For creating directories


class TAAF(nn.Module):
    def forward(self, x):
        numerator = torch.exp(-x)
        denominator = torch.exp(-x) + torch.exp(-(x**2))  # Sum of e^{-x} and e^{-x^2}
        return (numerator / denominator) - (1 / 2)  # TAAF formula


class TAAFConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        stride=1,
        norm=nn.GroupNorm,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            stride=stride,
            bias=False,
        )
        self.norm = norm(32, out_channels)  # GroupNorm with 32 groups is common
        self.taaf = TAAF()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.taaf(x)
        return x


class TAAFUpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        stride=1,
        norm=nn.GroupNorm,
        scale_factor=2,
    ):
        super().__init__()
        # Using ConvTranspose2d for upsampling
        self.upsample = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=scale_factor,
            stride=scale_factor,
            bias=False,
        )
        self.conv = nn.Conv2d(
            out_channels * 2,
            out_channels,
            kernel_size,
            padding=padding,
            stride=stride,
            bias=False,
        )  # *2 because of concatenation
        self.norm = norm(32, out_channels)
        self.taaf = TAAF()

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)  # Concatenate with skip connection
        x = self.conv(x)
        x = self.norm(x)
        x = self.taaf(x)
        return x


class ImageGenModelTAAF(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, num_levels=4):
        super().__init__()
        self.num_levels = num_levels
        channels = [base_channels * (2**i) for i in range(num_levels)]

        # Encoder path
        self.encoder_blocks = nn.ModuleList(
            [
                TAAFConvBlock(in_channels if i == 0 else channels[i - 1], channels[i])
                for i in range(num_levels)
            ]
        )
        self.downsample_convs = nn.ModuleList(
            [
                nn.Conv2d(channels[i], channels[i], kernel_size=3, stride=2, padding=1)
                for i in range(num_levels - 1)
            ]
        )

        # Bottleneck
        self.bottleneck_conv1 = TAAFConvBlock(channels[-1], channels[-1] * 2)
        self.bottleneck_conv2 = TAAFConvBlock(channels[-1] * 2, channels[-1])

        # Decoder path
        self.upsample_blocks = nn.ModuleList(
            [
                TAAFUpsampleBlock(channels[i + 1], channels[i])
                for i in reversed(range(num_levels - 1))
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                TAAFConvBlock(
                    channels[i] * 2, channels[i]
                )  # *2 because of concatenation
                for i in reversed(range(num_levels - 1))
            ]
        )
        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1)
        # --- Removed tanh activation ---
        # self.tanh = nn.Tanh()

    def forward(self, x):
        skips = []
        # Encoder
        for i in range(self.num_levels - 1):
            x = self.encoder_blocks[i](x)
            skips.append(x)
            x = self.downsample_convs[i](x)
        x = self.encoder_blocks[-1](x)  # Last encoder level without downsampling
        skips.append(x)

        # Bottleneck
        x = self.bottleneck_conv1(x)
        x = self.bottleneck_conv2(x)

        # Decoder
        for i in range(self.num_levels - 2, -1, -1):
            x = self.upsample_blocks[i](
                x, skips[i + 1]
            )  # Upsample and concatenate skip
            x = self.decoder_blocks[i](x)

        x = self.final_conv(x)
        # --- tanh is no longer applied here ---
        # x = self.tanh(x)
        return x


# Example usage:
if __name__ == "__main__":
    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_taaf = ImageGenModelTAAF().to(device)  # Move model to GPU

    # Print model summary (optional, requires torchinfo or similar)
    # from torchinfo import summary
    # print(summary(model_taaf.to(torch.device('cpu')), input_size=(1, 3, 256, 256))) # Move a copy to CPU for summary if needed

    # --- Load CIFAR10 dataset using torchvision.datasets ---
    dataset_path = "./data"  # Specify a path for dataset download
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)  # Create directory if it doesn't exist

    dataset = datasets.CIFAR10(
        root=dataset_path, train=True, download=True, transform=transforms.ToTensor()
    )  # Load CIFAR10, download to ./data_cifar10, and convert to Tensor

    # Preprocess function (adjust for torchvision CIFAR10 dataset)
    def preprocess_function(
        batch,
    ):  # batch is not directly used here as torchvision dataset is already loaded
        images = [
            img for img, label in dataset
        ]  # torchvision dataset yields (image, label) tuples, extract image
        # Resize images and normalize to [0, 1] (already done by ToTensor in dataset loading)
        resized_images = [
            transforms.Resize((256, 256))(img) for img in images
        ]  # Resize to 256x256
        return {
            "pixel_values": resized_images
        }  # Return as dictionary to match previous structure

    # Apply preprocessing (resize) -  Note: Normalization to [0, 1] is already done by `transforms.ToTensor()`
    processed_dataset = preprocess_function(
        dataset
    )  # Process the entire dataset at once for this example

    # Get a sample batch (for demonstration, using the first image)
    sample_batch = processed_dataset["pixel_values"][
        :1
    ]  # Get the first image from processed data
    if not isinstance(sample_batch, torch.Tensor):
        sample_batch = torch.stack(sample_batch)  # Stack list of tensors if needed
    sample_batch = sample_batch.unsqueeze(0).to(
        device
    )  # Add batch dimension and move to GPU

    print("Sample batch shape from dataset:", sample_batch.shape)

    # Pass the input through the model
    generated_image = model_taaf(sample_batch)

    print("Generated image shape:", generated_image.shape)
    print(
        "Generated image range (without tanh):",
        generated_image.min().item(),
        generated_image.max().item(),
    )  # Range will likely be different

    # --- Image Saving ---
    # Denormalize the generated image to the 0-1 range for saving (already in 0-1 range now)
    denormalized_image = generated_image.squeeze(0)  # Squeeze batch dimension

    # Ensure image is in CPU and save it using torchvision.utils.save_image
    save_path = "generated_image_taaf_gpu_torchvision.png"
    save_image(denormalized_image.cpu(), save_path)  # Save to file (e.g., PNG)
    print(f"Generated image saved to: {save_path}")
