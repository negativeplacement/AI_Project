import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMAGE_SIZE = 64
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    # --- Data Augmentation Steps ---
    transforms.RandomHorizontalFlip(p=0.5),     # 50% chance to flip image horizontally
    transforms.RandomRotation(degrees=15),       # Rotate randomly between -15 and +15 degrees
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Shift image slightly up/down/left/right
    transforms.ColorJitter(brightness=0.2, contrast=0.2),    # Randomly tweak lighting
    # -------------------------------
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize RGB channels
])
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # Must be the same size as training
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = datasets.ImageFolder(root='drive/MyDrive/dataset/train', transform=transform) #root should the the directory that has the folders not the folder dir
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

val_dataset = datasets.ImageFolder(root='drive/MyDrive/dataset/test', transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) # No need to shuffle validation data

class CNN(nn.Module):
    def __init__(self, num_layers=1):
        super(CNN, self).__init__()
        self.num_layers = num_layers # Initialize self.num_layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1) # 1. Convolutional Layer: 1 input channel (e.g., grayscale), 32 output channels, 3x3 filter
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 2. Max Pooling Layer: Reduces spatial dimensions (height/width) by half

        if self.num_layers == 2:
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(64 * 16 * 16, 128)
        else:
            self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)  # 10 output classes (e.g., digits 0-9)

    def forward(self, x):
        # Apply convolution, then ReLU activation, then pooling
        x = self.pool(F.relu(self.conv1(x)))
        if self.num_layers == 2:
            x = self.pool(F.relu(self.conv2(x)))

        # Flatten the multi-dimensional tensor into a 1D vector for the linear layers
        x = torch.flatten(x, start_dim=1)

        # Pass through the fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Testing the Network ---
def run_experiment(exp_name, num_layers, lr, epochs=5):
  print(f"\n=== Starting {exp_name}: Layers={num_layers}, LR={lr} ===")
  model = CNN(num_layers=num_layers) # Pass num_layers to the constructor
  criterion = nn.CrossEntropyLoss() # Define Loss Function (CrossEntropy is standard for classification)
  optimizer = optim.Adam(model.parameters(), lr=lr) # Define Adam Optimizer
  for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
      optimizer.zero_grad()             # Clear old gradients from the last step
      outputs = model(inputs)           # Forward pass: get predictions
      loss = criterion(outputs, labels) # Calculate error
      loss.backward()                   # Backward pass: calculate gradients
      optimizer.step()                  # Update network weights
      train_loss += loss.item()
    model.eval() # Set model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
      for inputs, labels in val_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")
  torch.save(model.state_dict(), f'cnn_{exp_name}_model_weights.pth')
  print(f"Model {exp_name} weights successfully saved!")

hyperparameters = [
    {"name": "Exp_1", "layers": 1, "lr": 0.001},  # l_1, lr_1
    {"name": "Exp_2", "layers": 1, "lr": 0.0001}, # l_1, lr_2
    {"name": "Exp_3", "layers": 2, "lr": 0.001},  # l_2, lr_1
    {"name": "Exp_4", "layers": 2, "lr": 0.0001}  # l_2, lr_2
]
for exp in hyperparameters:
  run_experiment(exp["name"], exp["layers"], exp["lr"], epochs=5)