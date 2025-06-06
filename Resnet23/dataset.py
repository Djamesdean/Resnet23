import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import CLASS_NAMES, IMAGE_SIZE


class Cifar_Custom(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for filename in os.listdir(class_dir):
                if filename.endswith(".png"):
                    self.image_paths.append(os.path.join(class_dir, filename))
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_train_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),               # Flip image horizontally
        transforms.RandomApply([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)         # Random brightness, contrast, saturation, hue
        ], p=0.5),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),  # Small rotation + translation
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize to [-1, 1]
    ])


def get_default_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])