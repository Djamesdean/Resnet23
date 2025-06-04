import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from Resnet23.config import BATCH_SIZE, DATA_DIR, NUM_CLASSES, RANDOM_SEED
from Resnet23.dataset import Cifar_Custom, get_default_transform
from Resnet23.modeling.Model23 import resnet23

torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load full training dataset
full_train_dataset = Cifar_Custom(root_dir=f"{DATA_DIR}/train", transform=get_default_transform())

# ✅ Split into training and validation sets
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# ✅ Loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load test set as before
test_dataset = Cifar_Custom(root_dir=f"{DATA_DIR}/test", transform=get_default_transform())
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = resnet23(num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch}: Train Loss={running_loss/total:.4f}, Accuracy={correct/total:.4f}")

# ✅ NEW: Validation function
def validate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Validation Accuracy={correct/total:.4f}")

def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy={correct/total:.4f}")

def main(num_epochs=5):
    for epoch in range(1, num_epochs + 1):
        train_one_epoch(epoch)
        validate()  # ✅ Run validation after each epoch
    evaluate()
    torch.save(model.state_dict(), "models/resnet23_cifar.pth")

if __name__ == "__main__":
    main()
