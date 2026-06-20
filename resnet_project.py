import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# --- Data Preparation ---
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train_dataset = datasets.ImageFolder('drive/MyDrive/dataset/train', transform=data_transforms['train'])
test_dataset = datasets.ImageFolder('drive/MyDrive/dataset/test', transform=data_transforms['test'])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
num_classes = len(train_dataset.classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_resnet_experiment(exp_name, fine_tune_strategy, lr, epochs=5):
  weights = models.ResNet18_Weights.DEFAULT
  model = models.resnet18(weights=weights)
  if fine_tune_strategy == "feature_extractor":
    # Freeze EVERYTHING initially
    for param in model.parameters():
      param.requires_grad = False
  elif fine_tune_strategy == "fine_tune_layer4":
    # Freeze everything EXCEPT the last layer block (layer4)
    for name, param in model.named_parameters():
      if "layer4" in name:
        param.requires_grad = True
      else:
        param.requires_grad = False
  # Replace the final classification head
  num_features = model.fc.in_features
  model.fc = nn.Linear(num_features, num_classes)
  model.fc.requires_grad = True
  model = model.to(device)
  criterion = nn.CrossEntropyLoss()
  params_to_update = [param for param in model.parameters() if param.requires_grad]
  optimizer = optim.Adam(params_to_update, lr=lr)
  for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad() # Reset gradients
        outputs = model(images) # Forward pass
        loss = criterion(outputs, labels)
        loss.backward() # Backward pass
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct_train / total_train
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad(): # Disable gradient math to save memory and speed up evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
    test_acc = correct_test / total_test

    print(f"Epoch [{epoch+1}/{epochs}] -> Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2%} -> Test Acc:  {test_acc:.2%}")


resnet_experiments = [
    {"name": "Exp_1", "strategy": "feature_extractor", "lr": 0.001},  # hp_1_1, hp_2_1
    {"name": "Exp_2", "strategy": "feature_extractor", "lr": 0.0001}, # hp_1_1, hp_2_2
    {"name": "Exp_3", "strategy": "fine_tune_layer4",   "lr": 0.001},  # hp_1_2, hp_2_1
    {"name": "Exp_4", "strategy": "fine_tune_layer4",   "lr": 0.0001}  # hp_1_2, hp_2_2
]

for exp in resnet_experiments:
    run_resnet_experiment(exp["name"], exp["strategy"], exp["lr"], epochs=5)