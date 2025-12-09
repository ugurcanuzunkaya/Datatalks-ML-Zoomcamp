import os
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import urllib.request

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def download_and_unzip_data():
    url = "https://github.com/SVizor42/ML_Zoomcamp/releases/download/straight-curly-data/data.zip"
    filename = "data.zip"
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)

    if not os.path.exists("data"):
        print(f"Unzipping {filename}...")
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(".")


def get_data_loaders(augment=False):
    train_transform_list = [
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    if augment:
        train_transform_list = [
            transforms.RandomRotation(50),
            transforms.RandomResizedCrop(200, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
        ] + train_transform_list

    train_transforms = transforms.Compose(train_transform_list)

    test_transforms = transforms.Compose(
        [
            transforms.Resize((200, 200)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = ImageFolder("./data/train", transform=train_transforms)
    test_dataset = ImageFolder("./data/test", transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=2)

    return train_loader, test_loader, len(train_dataset), len(test_dataset)


class HairNet(nn.Module):
    def __init__(self):
        super(HairNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()

        # Calculate size after conv and pool
        # Input: 200x200
        # Conv (3x3): 198x198
        # Pool (2x2): 99x99
        self.fc1 = nn.Linear(32 * 99 * 99, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(
    model,
    train_loader,
    test_loader,
    train_len,
    test_len,
    num_epochs=10,
    optimizer=None,
    criterion=None,
    device="cpu",
):
    history = {"acc": [], "loss": [], "val_acc": [], "val_loss": []}

    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.8)
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / train_len
        epoch_acc = correct_train / total_train
        history["loss"].append(epoch_loss)
        history["acc"].append(epoch_acc)

        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels = labels.float().unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_epoch_loss = val_running_loss / test_len
        val_epoch_acc = correct_val / total_val
        history["val_loss"].append(val_epoch_loss)
        history["val_acc"].append(val_epoch_acc)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, "
            f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}"
        )

    return history


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 1. Download Data
    download_and_unzip_data()

    # 2. Model Structure & Parameters
    model = HairNet().to(device)

    # Question 1: Loss function
    print("\nQuestion 1: Which loss function you will use?")
    print(
        "Answer: nn.BCEWithLogitsLoss()"
    )  # Since we used 1 output neuron and need binary classification

    # Question 2: Total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nQuestion 2: Total parameters: {total_params}")

    # 3. Training without augmentation
    print("\nStarting training (no augmentation)...")
    train_loader, test_loader, train_len, test_len = get_data_loaders(augment=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.8)

    history = train_model(
        model,
        train_loader,
        test_loader,
        train_len,
        test_len,
        num_epochs=10,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )

    # Question 3: Median of training accuracy
    median_acc = np.median(history["acc"])
    print(f"\nQuestion 3: Median of training accuracy: {median_acc:.2f}")

    # Question 4: Standard deviation of training loss
    std_loss = np.std(history["loss"])
    print(f"Question 4: Standard deviation of training loss: {std_loss:.3f}")

    # 4. Training with augmentation
    print("\nStarting training (with augmentation)...")
    # Note: We continue training the SAME model
    train_loader_aug, test_loader_aug, train_len_aug, test_len_aug = get_data_loaders(
        augment=True
    )

    history_aug = train_model(
        model,
        train_loader_aug,
        test_loader_aug,
        train_len_aug,
        test_len_aug,
        num_epochs=10,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )

    # Question 5: Mean of test loss for all epochs (trained with augmentations)
    mean_val_loss_aug = np.mean(history_aug["val_loss"])
    print(f"\nQuestion 5: Mean of test loss (augmented): {mean_val_loss_aug:.3f}")

    # Question 6: Average of test accuracy for the last 5 epochs (6 to 10)
    avg_val_acc_last_5 = np.mean(history_aug["val_acc"][5:])
    print(
        f"Question 6: Average of test accuracy for the last 5 epochs: {avg_val_acc_last_5:.2f}"
    )


if __name__ == "__main__":
    main()
