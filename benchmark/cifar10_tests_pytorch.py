import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# Define the CNN model matching your custom architecture.
class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()
        # Conv layer: 3 channels -> 16 channels, kernel=3x3, stride=1, padding=1 (keeps size 32x32)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        # Max Pooling: kernel=2, stride=2 reduces spatial dims: 32x32 -> 16x16
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Conv layer: 16 channels -> 32 channels; input size now 16x16; same padding maintains size.
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Second max pooling: 16x16 -> 8x8.
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layer: flatten 32*8*8 features to num_classes.
        self.fc = nn.Linear(32 * 8 * 8, num_classes)
        # Softmax layer over classes.
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)    # [batch, 16, 32, 32]
        x = self.pool1(x)    # [batch, 16, 16, 16]
        x = self.conv2(x)    # [batch, 32, 16, 16]
        x = self.pool2(x)    # [batch, 32, 8, 8]
        x = x.view(x.size(0), -1)  # flatten to [batch, 2048]
        x = self.fc(x)             # [batch, 10]
        x = self.softmax(x)        # output probabilities: softmax values
        return x

def main():
    # Set device (use CUDA if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transforms: Convert images to tensors with values in [0,1]
    transform = transforms.ToTensor()

    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='./data/cifar10/train', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data/cifar10/test', train=False, download=True, transform=transform)

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Instantiate the model, ensuring architecture matches the custom version.
    model = CustomCNN(num_classes=10).to(device)

    # Define optimizer (SGD with learning rate 0.01)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Because our model's forward pass includes softmax, we use NLLLoss with log-probabilities.
    # (Normally, one would remove the softmax layer and use CrossEntropyLoss.)
    criterion = nn.NLLLoss()

    num_epochs = 100

    # Training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)  # Output is probabilities from softmax.
            # Take log of the probabilities for NLLLoss.
            log_output = torch.log(output + 1e-8)  # Adding epsilon for numerical stability.
            loss = criterion(log_output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")

        # Optionally, print a sample prediction every few epochs.
        if epoch % 10 == 0:
            model.eval()
            sample_data, _ = next(iter(train_loader))
            sample_data = sample_data.to(device)
            sample_pred = model(sample_data)
            # print(f"Sample predictions at epoch {epoch}:")
            # Move to CPU for printing.
            # print(sample_pred.cpu().data)
    elapsed_time = time.time() - start_time
    print(f"PyTorch training completed in {elapsed_time:.2f} seconds")

    # Evaluate on test set.
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)
    accuracy = 100. * correct / total
    print(f"PyTorch Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()