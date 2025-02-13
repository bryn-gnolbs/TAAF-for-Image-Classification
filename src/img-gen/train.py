import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from tqdm import tqdm


# --- Define TAAF Activation (as provided) ---
class TAAF(nn.Module):
    def forward(self, x):
        numerator = torch.exp(-x)
        denominator = torch.exp(-x) + torch.exp(-(x**2))  # Sum of e^{-x} and e^{-x^2}
        return (numerator / denominator) - (1 / 2)  # TAAF formula


# --- Define Generator Network ---
class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels, features_g=64):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.net = nn.Sequential(
            # Input: latent_dim x 1 x 1
            self._block(
                latent_dim, features_g * 16, 4, 1, 0
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

    def forward(self, z):
        return self.net(z.reshape(-1, self.latent_dim, 1, 1))


# --- Define Discriminator Network ---
class Discriminator(nn.Module):
    def __init__(self, img_channels, features_d=64):
        super(Discriminator, self).__init__()
        self.img_channels = img_channels
        self.net = nn.Sequential(
            # Input: img_channels x 32 x 32
            nn.Conv2d(
                img_channels, features_d, kernel_size=4, stride=2, padding=1
            ),  # Output: features_d x 16 x 16
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

    def forward(self, x):
        return self.net(x).reshape(-1, 1)


# --- Hyperparameters and Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4  # Consistent with DCGAN papers
BATCH_SIZE = 128
IMAGE_SIZE = 32  # CIFAR-10 images are 32x32
IMG_CHANNELS = 3  # CIFAR-10 color images
LATENT_DIM = 128  # Size of the noise vector
NUM_EPOCHS = 100  # You can increase this
FEATURES_DISC = 64
FEATURES_GEN = 64

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

# --- Initialize Generator and Discriminator ---
generator = Generator(LATENT_DIM, IMG_CHANNELS, FEATURES_GEN).to(device)
discriminator = Discriminator(IMG_CHANNELS, FEATURES_DISC).to(device)


# Initialize weights (DCGAN paper suggests using normal distribution)
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)  # mean=0, std=0.02


generator.apply(initialize_weights)
discriminator.apply(initialize_weights)


# --- Optimizers ---
optimizer_gen = optim.AdamW(
    generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)
)  # betas are standard for DCGAN
optimizer_disc = optim.AdamW(
    discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)
)

# --- Loss Function ---
criterion = nn.BCELoss()

# --- Fixed noise for image generation during training ---
fixed_noise = torch.randn(32, LATENT_DIM, device=device)  # Generate 32 sample images

# --- Directories for saving images and models ---
sample_dir = "./tests/gen-img/images"
checkpoint_dir = "./tests/gen-img"
os.makedirs(sample_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)


# --- Training Loop ---
def train_dcgan(
    dataloader,
    generator,
    discriminator,
    optimizer_gen,
    optimizer_disc,
    criterion,
    num_epochs,
    fixed_noise,
    sample_dir,
    checkpoint_dir,
):
    generator.train()
    discriminator.train()

    for epoch in range(num_epochs):
        for batch_idx, (real_images, _) in enumerate(
            tqdm(dataloader)
        ):  # _ because we don't need labels for GANs
            real_images = real_images.to(device)
            batch_size = real_images.shape[0]

            # --- Train Discriminator: maximize log(D(x)) + log(1 - D(G(z))) ---
            noise = torch.randn(batch_size, LATENT_DIM, device=device)
            fake_images = generator(noise)

            disc_real = discriminator(real_images).reshape(-1)
            loss_disc_real = criterion(
                disc_real, torch.ones_like(disc_real)
            )  # Target for real images is 1

            disc_fake = discriminator(fake_images.detach()).reshape(
                -1
            )  # Detach to avoid backprop to generator in disc step
            loss_disc_fake = criterion(
                disc_fake, torch.zeros_like(disc_fake)
            )  # Target for fake images is 0

            loss_disc = (loss_disc_real + loss_disc_fake) / 2

            discriminator.zero_grad()
            loss_disc.backward()
            optimizer_disc.step()

            # --- Train Generator: maximize log(D(G(z))) ---
            output = discriminator(fake_images).reshape(-1)
            loss_gen = criterion(
                output, torch.ones_like(output)
            )  # Target for fake images to be considered real is 1

            generator.zero_grad()
            loss_gen.backward()
            optimizer_gen.step()

            # --- Print progress and save samples ---
            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(dataloader)} \
                      Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}"
                )

                with torch.no_grad():
                    fake_samples = generator(fixed_noise)
                    save_image(
                        fake_samples[:32],
                        os.path.join(
                            sample_dir, f"epoch_{epoch}_batch_{batch_idx}.png"
                        ),
                        normalize=True,
                    )

        # --- Save model checkpoints ---
        if epoch % 5 == 0:
            torch.save(
                generator.state_dict(),
                os.path.join(checkpoint_dir, f"generator_epoch_{epoch}.pth"),
            )
            torch.save(
                discriminator.state_dict(),
                os.path.join(checkpoint_dir, f"discriminator_epoch_{epoch}.pth"),
            )
            print(f"Saved checkpoints for epoch {epoch}")

    print("Training finished!")


# --- Start Training ---
train_dcgan(
    dataloader,
    generator,
    discriminator,
    optimizer_gen,
    optimizer_disc,
    criterion,
    NUM_EPOCHS,
    fixed_noise,
    sample_dir,
    checkpoint_dir,
)

print("Sample images are saved in:", sample_dir)
print("Model checkpoints are saved in:", checkpoint_dir)
