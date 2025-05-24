import sys

from PIL import Image
import torch
from torchvision import transforms

from Resnet23.config import CLASS_NAMES, IMAGE_SIZE, NUM_CLASSES
from Resnet23.resnet23 import resnet23

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = resnet23(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load("resnet23_cifar.pth", map_location=device))
model.to(device)
model.eval()

# Define the transform to apply to input images
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
        class_name = CLASS_NAMES[class_idx]
        print(f"Predicted class: {class_name}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path_to_image")
        sys.exit(1)
    predict(sys.argv[1])
