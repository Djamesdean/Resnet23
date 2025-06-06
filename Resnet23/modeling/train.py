import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from Resnet23.config import BATCH_SIZE, DATA_DIR, NUM_CLASSES, RANDOM_SEED, setup_experiment
from Resnet23.dataset import Cifar_Custom, get_default_transform, get_train_transform
from Resnet23.modeling.matrics import evaluate_metrics
from Resnet23.modeling.Model23 import resnet23

torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load full training dataset
full_train_dataset = Cifar_Custom(root_dir=f"{DATA_DIR}/train", transform=None)

# Split into training and validation sets
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Apply transforms separately
train_dataset.dataset.transform = get_train_transform()      
val_dataset.dataset.transform = get_default_transform() 

# ✅ Loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dataset = Cifar_Custom(root_dir=f"{DATA_DIR}/test", transform=get_default_transform())
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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

    train_loss = running_loss / total
    train_acc = correct / total
    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Accuracy={train_acc:.4f}")
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("train_accuracy", train_acc, step=epoch)

# NEW: Validation function
def validate(epoch):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0) 
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= total
    val_acc = correct / total
    print(f"Validation Accuracy={val_acc:.4f}")
    mlflow.log_metric("val_loss", val_loss, step=epoch)
    mlflow.log_metric("val_accuracy", val_acc, step=epoch)

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
    test_acc = correct / total
    print(f"Test Accuracy={test_acc:.4f}")
    mlflow.log_metric("test_accuracy", test_acc)

def main(num_epochs=10):
    
    logger = setup_experiment()
    if not logger:
        print("❌ Failed to setup experiment. Exiting...")
        return

    with mlflow.start_run():
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("epochs", num_epochs)
        mlflow.log_param("model", "resnet23")
        mlflow.log_param("dataset", "CIFAR-10")

        for epoch in range(1, num_epochs + 1):
            train_one_epoch(epoch)
            validate(epoch)

        evaluate()

        evaluate_metrics(model, test_loader, device, NUM_CLASSES)
        
        model_path = "models/resnet23_cifar.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        mlflow.pytorch.log_model(model, "resnet23_model")
        
        print("Training complete and model saved.")

if __name__ == "__main__":
    main()
