import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import tempfile  # Import for temporary directories
from tqdm import tqdm
from pytorch_fid import fid_score  # Import FID calculation


# --- Define TAAF Activation (as provided) ---
class TAAF(nn.Module):
    def forward(self, x):
        numerator = torch.exp(-x)
        denominator = torch.exp(-x) + torch.exp(-(x**2))  # Sum of e^{-x} and e^{-x^2}
        return (numerator / denominator) - (1 / 2)  # TAAF formula


# --- Define Conditional Generator Network ---
class ConditionalGenerator(nn.Module):
    def __init__(
        self, latent_dim, img_channels, num_classes, embedding_dim, features_g=64
    ):
        super(ConditionalGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)

        self.net = nn.Sequential(
            # Input: (latent_dim + embedding_dim) x 1 x 1
            self._block(
                latent_dim + embedding_dim, features_g * 16, 4, 1, 0
            ),  # Output: (features_g*16) x 4 x 4
            self._block(
                features_g * 16, features_g * 8, 4, 2, 1
            ),  # Output: (features_g*8) x 8 x 8
            self._block(
                features_g * 8, features_g * 4, 4, 2, 1
            ),  # Output: (features_g*4) x 16 x 16
            self._block(
                features_g * 4, features_g * 2, 4, 2, 1
            ),  # Output: (features_g*2) x 32 x 32
            nn.ConvTranspose2d(
                features_g * 2, img_channels, kernel_size=4, stride=2, padding=1
            ),  # Output: img_channels x 64 x 64 (oops should be 32x32 for cifar10, will fix kernel size or remove layer later)
            nn.Tanh(),  # Output should be in range [-1, 1] for images
        )
        # Correcting the last layer for CIFAR-10 (32x32 output)
        self.net[-2] = nn.ConvTranspose2d(
            features_g * 2, img_channels, kernel_size=4, stride=2, padding=1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            TAAF(),  # Using TAAF activation here
        )

    def forward(self, z, labels):
        # Convert label to embedding
        label_embedding = self.label_embedding(labels)
        # Concatenate latent noise and label embedding
        combined_input = torch.cat(
            (z, label_embedding), dim=1
        )  # Concatenate along channel dimension
        return self.net(
            combined_input.reshape(-1, self.latent_dim + self.embedding_dim, 1, 1)
        )


# --- Define Conditional Discriminator Network ---
class ConditionalDiscriminator(nn.Module):
    def __init__(self, img_channels, num_classes, embedding_dim, features_d=64):
        super(ConditionalDiscriminator, self).__init__()
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)

        self.net = nn.Sequential(
            # Input: img_channels x 32 x 32
            nn.Conv2d(
                img_channels + embedding_dim,
                features_d,
                kernel_size=4,
                stride=2,
                padding=1,
            ),  # Output: features_d x 16 x 16 (input channel is now image channels + embedding dim)
            TAAF(),  # Using TAAF activation here
            self._block(
                features_d, features_d * 2, 4, 2, 1
            ),  # Output: (features_d*2) x 8 x 8
            self._block(
                features_d * 2, features_d * 4, 4, 2, 1
            ),  # Output: (features_d*4) x 4 x 4
            self._block(
                features_d * 4, features_d * 8, 4, 2, 1
            ),  # Output: (features_d*8) x 2 x 2
            nn.Conv2d(
                features_d * 8, 1, kernel_size=2, stride=1, padding=0
            ),  # Output: 1 x 1 x 1
            nn.Sigmoid(),  # Output probability [0, 1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            TAAF(),  # Using TAAF activation here
        )

    def forward(self, x, labels):
        # Convert label to embedding
        label_embedding = self.label_embedding(labels)
        # Replicate label embedding spatially to match image size (for concatenation)
        label_embedding_resized = (
            label_embedding.unsqueeze(2).unsqueeze(3).repeat(1, 1, x.size(2), x.size(3))
        )
        # Concatenate image and label embedding along channel dimension
        combined_input = torch.cat((x, label_embedding_resized), dim=1)
        return self.net(combined_input).reshape(-1, 1)


# --- Hyperparameters and Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4  # Consistent with DCGAN papers
BATCH_SIZE = 128
IMAGE_SIZE = 32  # CIFAR-10 images are 32x32
IMG_CHANNELS = 3  # CIFAR-10 color images
LATENT_DIM = 128  # Size of the noise vector
NUM_EPOCHS = 100  # You can increase this
FEATURES_DISC = 64
FEATURES_GEN = 128  # Increased Generator features
NUM_CLASSES = 10  # CIFAR-10 has 10 classes
EMBEDDING_DIM = 256  # Increased Embedding Dimension

# --- Data Loading ---
transforms_ = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),  # Ensure images are 32x32
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        ),  # Normalize to [-1, 1] range, good for Tanh output
    ]
)

dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transforms_
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# --- Get a fixed batch of real images for FID calculation ---
real_batch_for_fid = next(iter(dataloader))[0][:32].to(
    device
)  # Use first 32 real images for FID, keep on GPU

# --- Initialize Conditional Generator and Discriminator ---
generator = ConditionalGenerator(
    LATENT_DIM, IMG_CHANNELS, NUM_CLASSES, EMBEDDING_DIM, FEATURES_GEN
).to(device)
discriminator = ConditionalDiscriminator(
    IMG_CHANNELS, NUM_CLASSES, EMBEDDING_DIM, FEATURES_DISC
).to(device)


# Initialize weights (DCGAN paper suggests using normal distribution)
def initialize_weights(model):
    for m in model.modules():
        if isinstance(
            m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.Embedding)
        ):  # Include Embedding layer
            nn.init.normal_(m.weight.data, 0.0, 0.02)  # mean=0, std=0.02


generator.apply(initialize_weights)
discriminator.apply(initialize_weights)


# --- Optimizers ---
optimizer_gen = optim.Adam(
    generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)
)  # betas are standard for DCGAN
optimizer_disc = optim.Adam(
    discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)
)

# --- Loss Function ---
criterion = nn.BCELoss()

# --- Fixed noise and labels for image generation during training ---
fixed_noise = torch.randn(
    NUM_CLASSES, LATENT_DIM, device=device
)  # Generate NUM_CLASSES samples
fixed_labels = torch.arange(0, NUM_CLASSES, device=device)  # Labels 0 to NUM_CLASSES-1

# --- Directories for saving images and models ---
sample_dir = "./tests/gen-img/images"
checkpoint_dir = "./tests/gen-img"
os.makedirs(sample_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

CIFAR10_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# --- Function to preprocess images for FID calculation ---
def preprocess_image_for_fid(images):
    # Rescale from [-1, 1] to [0, 255] and convert to uint8
    return (
        ((images * 0.5 + 0.5) * 255).clamp(0, 255).to(torch.uint8).cpu()
    )  # Move to CPU and uint8


# --- Training Loop ---
def train_conditional_dcgan(
    dataloader,
    generator,
    discriminator,
    optimizer_gen,
    optimizer_disc,
    criterion,
    num_epochs,
    fixed_noise,
    fixed_labels,
    sample_dir,
    checkpoint_dir,
    real_batch_for_fid,  # Pass fixed real batch
):
    generator.train()
    discriminator.train()

    # --- Prepare real images for FID calculation ---
    real_images_for_fid = preprocess_image_for_fid(real_batch_for_fid)

    for epoch in range(num_epochs):
        for batch_idx, (real_images, labels) in enumerate(tqdm(dataloader)):
            real_images = real_images.to(device)
            labels = labels.to(device)  # Class labels as conditions
            batch_size = real_images.shape[0]

            # --- Train Discriminator: maximize log(D(x, y)) + log(1 - D(G(z, y), y)) ---
            noise = torch.randn(batch_size, LATENT_DIM, device=device)
            fake_images = generator(noise, labels)  # Generator now takes labels

            disc_real = discriminator(real_images, labels).reshape(
                -1
            )  # Discriminator now takes labels
            loss_disc_real = criterion(
                disc_real, torch.ones_like(disc_real)
            )  # Target for real images is 1

            disc_fake = discriminator(
                fake_images.detach(), labels
            ).reshape(  # Discriminator now takes labels
                -1
            )  # Detach to avoid backprop to generator in disc step
            loss_disc_fake = criterion(
                disc_fake, torch.zeros_like(disc_fake)
            )  # Target for fake images is 0

            loss_disc = (loss_disc_real + loss_disc_fake) / 2

            discriminator.zero_grad()
            loss_disc.backward()
            optimizer_disc.step()

            # --- Train Generator: maximize log(D(G(z, y), y)) ---
            output = discriminator(fake_images, labels).reshape(
                -1
            )  # Discriminator now takes labels
            loss_gen = criterion(
                output, torch.ones_like(output)
            )  # Target for fake images to be considered real is 1

            generator.zero_grad()
            loss_gen.backward()
            optimizer_gen.step()

            # --- Print progress, save samples and calculate FID ---
            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(dataloader)} \
                      Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}"
                )

                with torch.no_grad():
                    fake_samples = generator(
                        fixed_noise, fixed_labels
                    )  # Generate with fixed noise and labels
                    fake_images_fid = preprocess_image_for_fid(fake_samples)

                    # --- Create temporary directories ---
                    with (
                        tempfile.TemporaryDirectory() as real_temp_dir,
                        tempfile.TemporaryDirectory() as fake_temp_dir,
                    ):
                        # --- Save real images to temporary directory ---
                        for i in range(real_images_for_fid.size(0)):
                            save_image(
                                real_images_for_fid[i].float() / 255.0,
                                os.path.join(real_temp_dir, f"real_{i}.png"),
                            )  # Need to rescale back to [0, 1] for save_image

                        # --- Save fake images to temporary directory ---
                        for i in range(fake_images_fid.size(0)):
                            save_image(
                                fake_images_fid[i].float() / 255.0,
                                os.path.join(fake_temp_dir, f"fake_{i}.png"),
                            )  # Need to rescale back to [0, 1] for save_image

                        # --- Calculate FID using directory paths ---
                        fid_value = fid_score.calculate_fid_given_paths(
                            [real_temp_dir, fake_temp_dir],
                            batch_size=32,
                            device=device,
                            dims=2048,  # Added dims argument
                            num_workers=0,  # Setting num_workers to 0 to avoid multiprocessing issues for now
                        )
                        print(f"FID: {fid_value:.4f}")

                    # Save images with labels as filenames
                    for i in range(NUM_CLASSES):
                        save_image(
                            fake_samples[i],
                            os.path.join(
                                sample_dir,
                                f"epoch_{epoch}_batch_{batch_idx}_class_{CIFAR10_CLASS_NAMES[i]}.png",
                            ),
                            normalize=True,
                        )

        # --- Save model checkpoints ---
        if epoch % 5 == 0:
            torch.save(
                generator.state_dict(),
                os.path.join(
                    checkpoint_dir, f"conditional_generator_epoch_{epoch}.pth"
                ),
            )
            torch.save(
                discriminator.state_dict(),
                os.path.join(
                    checkpoint_dir, f"conditional_discriminator_epoch_{epoch}.pth"
                ),
            )
            print(f"Saved conditional checkpoints for epoch {epoch}")

    print("Conditional Training finished!")


# --- Start Conditional Training with FID ---
train_conditional_dcgan(
    dataloader,
    generator,
    discriminator,
    optimizer_gen,
    optimizer_disc,
    criterion,
    NUM_EPOCHS,
    fixed_noise,
    fixed_labels,
    sample_dir,
    checkpoint_dir,
    real_batch_for_fid,  # Pass fixed real batch
)

print("Conditional sample images are saved in:", sample_dir)
print("Conditional model checkpoints are saved in:", checkpoint_dir)
print("Make sure to install pytorch-fid: pip install pytorch-fid")
