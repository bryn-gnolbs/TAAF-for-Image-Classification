import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import threading
import time
import os


# --- Activation Functions ---
# Define The Analog Activation Function (TAAF)
class TAAF(nn.Module):
    def forward(self, x):
        numerator = torch.exp(-x)
        denominator = torch.exp(-x) + torch.exp(-(x**2))
        return (numerator / denominator) - (1 / 2)


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
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.beta = nn.Parameter(torch.tensor(float(beta)))

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


# --- CIFAR10 Model with Activation Choice ---
class CIFAR10TAAF(nn.Module):
    def __init__(
        self,
        activation_type="TAAF",
        elu_alpha=1.0,
        elu_beta=1.0,
        extaaf_alpha=1.0,
        extaaf_beta=1.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)

        if activation_type == "TAAF":
            self.activation = TAAF()
        elif activation_type == "ExTAAF":
            self.activation = ExTAAF(alpha=extaaf_alpha, beta=extaaf_beta)
        elif activation_type == "ELU":
            self.activation = ELU(alpha=elu_alpha, beta=elu_beta)
        elif activation_type == "DISEM10":
            self.activation = DISEM10()
        elif activation_type == "ReLU":
            self.activation = nn.ReLU()
        elif activation_type == "Tanh":
            self.activation = nn.Tanh()
        elif activation_type == "Sigmoid":
            self.activation = nn.Sigmoid()
        elif activation_type == "GELU":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Activation type '{activation_type}' not recognized.")

    def forward(self, x):
        # Keep feature maps for visualization
        features = {}
        features["input"] = x.clone()

        x = self.conv1(x)
        features["conv1"] = x.clone()
        x = self.activation(x)
        features["conv1_act"] = x.clone()
        x = self.pool(x)
        features["conv1_pool"] = x.clone()

        x = self.conv2(x)
        features["conv2"] = x.clone()
        x = self.activation(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.activation(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        features["fc1"] = x.clone()

        x = self.fc2(x)
        features["output"] = x.clone()

        return x, features


# --- ELC Calculation Function ---
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
                f_x_i = model(x_i)[
                    0
                ].sum()  # Get model output for x_i, sum to get scalar

                x_i_perturbed = x_i + delta * torch.randn_like(x_i)  # Perturb input
                x_i_perturbed = x_i_perturbed.to(device)
                f_x_i_delta = model(x_i_perturbed)[
                    0
                ].sum()  # Get model output for perturbed x_i

                elc_sample = torch.abs(f_x_i_delta - f_x_i) / delta
                total_elc += elc_sample.item()
                num_samples += 1
    return total_elc / num_samples if num_samples > 0 else 0.0


# UI and visualization class (remains largely the same)
class LiveNetworkVisualizer:
    def __init__(self):
        # Create main window
        self.root = tk.Tk()
        self.root.title("TAAF Neural Network Visualizer")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Set up figures
        self.fig_metrics = Figure(figsize=(8, 3))
        self.fig_samples = Figure(figsize=(8, 6))

        # Add figures to UI
        self.canvas_metrics = FigureCanvasTkAgg(self.fig_metrics, master=self.root)
        self.canvas_metrics.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.canvas_samples = FigureCanvasTkAgg(self.fig_samples, master=self.root)
        self.canvas_samples.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Initialize plots
        self.ax_loss = self.fig_metrics.add_subplot(121)
        self.ax_acc = self.fig_metrics.add_subplot(122)

        # Data storage
        self.iterations = []
        self.train_losses = []
        self.test_losses = []
        self.train_accs = []
        self.test_accs = []
        self.elc_values = []  # Store ELC values

        # Initialize lines
        (self.loss_line_train,) = self.ax_loss.plot([], [], "r-", label="Train Loss")
        (self.loss_line_test,) = self.ax_loss.plot([], [], "b-", label="Val Loss")
        (self.acc_line_train,) = self.ax_acc.plot([], [], "r-", label="Train Acc")
        (self.acc_line_test,) = self.ax_acc.plot([], [], "b-", label="Val Acc")
        (self.elc_line,) = self.ax_loss.plot([], [], "g-", label="ELC")  # Add ELC line

        self.ax_loss.set_title("Loss & ELC")  # Update title to include ELC
        self.ax_loss.set_xlabel("Iteration")
        self.ax_loss.legend()

        self.ax_acc.set_title("Accuracy")
        self.ax_acc.set_xlabel("Iteration")
        self.ax_acc.legend()

        self.fig_metrics.tight_layout()

        # Variables for training
        self.running = True
        self.classes = (
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
        )

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Initializing...")
        self.status_bar = tk.Label(
            self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def on_closing(self):
        self.running = False
        self.root.destroy()

    def update_metrics(
        self, iteration, train_loss, test_loss, train_acc, test_acc, elc_value
    ):
        self.iterations.append(iteration)
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.train_accs.append(train_acc)
        self.test_accs.append(test_acc)
        self.elc_values.append(elc_value)  # Append ELC value

        # Update lines
        self.loss_line_train.set_data(self.iterations, self.train_losses)
        self.loss_line_test.set_data(self.iterations, self.test_losses)
        self.acc_line_train.set_data(self.iterations, self.train_accs)
        self.acc_line_test.set_data(self.iterations, self.test_accs)
        self.elc_line.set_data(self.iterations, self.elc_values)  # Update ELC line

        # Adjust limits
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()
        self.ax_acc.relim()
        self.ax_acc.autoscale_view()

        # Update canvas
        self.canvas_metrics.draw()

    def update_sample_visualizations(self, images, labels, predictions, features):
        self.fig_samples.clear()

        # Sample image with prediction
        for i in range(min(4, len(images))):
            # Original image
            ax1 = self.fig_samples.add_subplot(3, 4, i + 1)
            img = images[i].permute(1, 2, 0).cpu().numpy()
            img = (
                img * torch.tensor([0.2023, 0.1994, 0.2010]).numpy()
                + torch.tensor([0.4914, 0.4822, 0.4465]).numpy()
            )
            img = torch.clamp(torch.tensor(img), 0, 1).numpy()
            ax1.imshow(img)

            # Set color based on prediction correctness
            color = "green" if predictions[i] == labels[i] else "red"
            ax1.set_title(
                f"Pred: {self.classes[predictions[i]]}\nTrue: {self.classes[labels[i]]}",
                color=color,
                fontsize=8,
            )
            ax1.axis("off")

            # Feature map visualization (first channel)
            if "conv1_act" in features:
                feature_map = features["conv1_act"][i, 0].detach().cpu()
                ax2 = self.fig_samples.add_subplot(3, 4, i + 5)
                ax2.imshow(feature_map, cmap="viridis")
                ax2.set_title(f"Conv1 Feature", fontsize=8)
                ax2.axis("off")

            # Final layer activations as bar chart
            if "output" in features:
                ax3 = self.fig_samples.add_subplot(3, 4, i + 9)
                output = features["output"][i].detach().cpu()
                ax3.bar(range(10), torch.softmax(output, dim=0).numpy())
                ax3.set_title("Class Probabilities", fontsize=8)
                ax3.set_xticks(range(10))
                ax3.set_xticklabels(range(10), fontsize=6)
                ax3.axvline(x=labels[i], color="blue", linestyle="--", linewidth=1)
                ax3.axvline(x=predictions[i], color="red", linestyle="--", linewidth=1)

        self.fig_samples.tight_layout()
        self.canvas_samples.draw()

    def update_status(self, text):
        self.status_var.set(text)
        self.root.update()


# Data and model setup functions (load_data remains the same)
def load_data(batch_size=64):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
            ),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
            ),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, test_loader


# Main training function with ELC and Activation Choice
def train_model(
    visualizer, activation_type, elu_alpha, elu_beta, extaaf_alpha, extaaf_beta
):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visualizer.update_status(f"Using device: {device}")

    # Setup model and optimizer
    model = CIFAR10TAAF(
        activation_type=activation_type,
        elu_alpha=elu_alpha,
        elu_beta=elu_beta,
        extaaf_alpha=extaaf_alpha,
        extaaf_beta=extaaf_beta,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # Create dirs
    os.makedirs("results", exist_ok=True)

    # Load data
    visualizer.update_status("Loading CIFAR-10 dataset...")
    train_loader, test_loader = load_data(batch_size=64)

    # Enable cuDNN benchmarking
    torch.backends.cudnn.benchmark = True

    # Initialize variables
    num_epochs = 20
    log_interval = 20  # Log every 20 batches
    iteration = 0
    visualize_interval = 10  # Visualize every 10 batches
    elc_interval = 50  # Calculate ELC every 50 batches
    elc_delta = 0.01  # Delta for ELC calculation

    # Create ELC data loader subset (using a smaller batch size for ELC calculation)
    elc_train_loader_subset = DataLoader(
        torch.utils.data.Subset(
            load_data(batch_size=len(train_loader.dataset))[0].dataset,
            range(0, len(train_loader.dataset), 100),
        ),  # Use every 100th sample for ELC
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    # Get a batch for visualization
    vis_data = next(iter(test_loader))
    vis_images, vis_labels = vis_data[0].to(device), vis_data[1].to(device)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        elc_value = 0  # Initialize ELC value for each epoch

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass with features
            optimizer.zero_grad()
            outputs, features = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update running statistics
            train_loss += loss.item()

            # Calculate ELC periodically
            if (
                batch_idx % elc_interval == 0 and batch_idx > 0
            ):  # Calculate ELC less frequently
                elc_value = calculate_elc(
                    model, elc_train_loader_subset, elc_delta, device
                )

            # Log progress frequently
            if batch_idx % log_interval == 0:
                # Calculate current metrics
                current_loss = train_loss / (batch_idx + 1)
                current_acc = 100.0 * correct / total

                # Run a quick validation sample
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    # Use a small subset of test data for quick validation
                    for i, (val_inputs, val_targets) in enumerate(test_loader):
                        if i >= 5:  # Just use a few batches for quick validation
                            break
                        val_inputs, val_targets = (
                            val_inputs.to(device),
                            val_targets.to(device),
                        )
                        val_outputs, _ = model(val_inputs)
                        loss = criterion(val_outputs, val_targets)

                        val_loss += loss.item()
                        _, val_predicted = val_outputs.max(1)
                        val_total += val_targets.size(0)
                        val_correct += val_predicted.eq(val_targets).sum().item()

                val_loss = val_loss / 5  # 5 batches
                val_acc = 100.0 * val_correct / val_total

                # Update metrics visualization
                iteration += 1
                visualizer.update_metrics(
                    iteration,
                    current_loss,
                    val_loss,
                    current_acc,
                    val_acc,
                    elc_value,  # Pass ELC value
                )

                # Update status
                status_text = f"Epoch: {epoch}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, "
                status_text += f"Loss: {current_loss:.4f}, Acc: {current_acc:.2f}%, "
                status_text += f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                status_text += f"ELC: {elc_value:.4f}"  # Display ELC in status
                visualizer.update_status(status_text)

                # Return to training mode
                model.train()

            # Update sample visualization frequently
            if batch_idx % visualize_interval == 0:
                model.eval()
                with torch.no_grad():
                    # Process visualization batch
                    vis_outputs, vis_features = model(vis_images[:4])
                    _, vis_preds = vis_outputs.max(1)

                    # Update visualization
                    visualizer.update_sample_visualizations(
                        vis_images[:4], vis_labels[:4], vis_preds[:4], vis_features
                    )
                model.train()

            # Check if we should continue
            if not visualizer.running:
                visualizer.update_status("Training interrupted.")
                return

        # End of epoch
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        # Calculate epoch metrics
        test_loss /= len(test_loader)
        test_acc = 100.0 * correct / total

        # Update scheduler
        scheduler.step()

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": test_loss,
                "val_acc": test_acc,
                "elc_value": elc_value,  # Save ELC in checkpoint
            },
            f"results/taaf_checkpoint_epoch_{epoch}.pth",
        )

        # Update status
        visualizer.update_status(
            f"Epoch {epoch} completed. Test Acc: {test_acc:.2f}%, ELC: {elc_value:.4f}"
        )  # Display ELC at epoch end

    # Save final model
    torch.save(model.state_dict(), "results/taaf_final_model.pth")
    visualizer.update_status("Training completed. Model saved.")


# Main entry point
def main():
    # --- Configuration ---
    activation_type = "ExTAAF"  # Choose activation type: TAAF, ExTAAF, ELU, DISEM10, ReLU, Tanh, Sigmoid, GELU
    elu_alpha = 1.0
    elu_beta = 1.0
    extaaf_alpha = 1.0
    extaaf_beta = 1.0
    # --- End Configuration ---

    # Create visualizer
    visualizer = LiveNetworkVisualizer()

    # Start training in a separate thread, passing activation type and params
    training_thread = threading.Thread(
        target=train_model,
        args=(
            visualizer,
            activation_type,
            elu_alpha,
            elu_beta,
            extaaf_alpha,
            extaaf_beta,
        ),
    )
    training_thread.daemon = True
    training_thread.start()

    # Start the UI loop
    visualizer.root.mainloop()


if __name__ == "__main__":
    main()
