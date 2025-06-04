import os

from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from .config import CLASS_NAMES, DATA_DIR, IMAGE_SIZE


def downsample_and_prepare_cifar10(output_dir=DATA_DIR, max_per_class=500):
 
    to_pil = ToPILImage()

    def save_split(dataset, split):
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        counter = {label: 0 for label in range(len(CLASS_NAMES))}

        for img, label in tqdm(dataset, desc=f"Saving {split}"):
            if counter[label] >= max_per_class:
                continue

            class_name = CLASS_NAMES[label]
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            img_path = os.path.join(class_dir, f"{counter[label]}.png")
            resized_img = transforms.Resize(IMAGE_SIZE)(img)
            resized_img.save(img_path)
            counter[label] += 1

    train_data = CIFAR10(root='data/raw', train=True, download=True)
    test_data = CIFAR10(root='data/raw', train=False, download=True)
    save_split(train_data, "train")
    save_split(test_data, "test")

if __name__ == "__main__":
    downsample_and_prepare_cifar10()