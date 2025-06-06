import os

from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from .config import CLASS_NAMES, DATA_DIR, IMAGE_SIZE


def downsample_and_prepare_cifar10(output_dir=DATA_DIR, max_per_class_train=500, max_per_class_test=100):
    def save_split(dataset, split, max_per_class):
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
    
        counter = {i: 0 for i in range(len(CLASS_NAMES))}
        total_saved = 0
        target_total = max_per_class * len(CLASS_NAMES)
        
        
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
        
        with tqdm(total=target_total, desc=f"Saving {split}") as pbar:
            for img, label in dataset:
                label = int(label)
                if counter[label] >= max_per_class:
                    continue
                class_name = CLASS_NAMES[label]
                class_dir = os.path.join(split_dir, class_name)
                img = transforms.Resize(IMAGE_SIZE)(img)
                img_path = os.path.join(class_dir, f"{counter[label]:04d}.png")
                img.save(img_path)
                counter[label] += 1
                total_saved += 1
                pbar.update(1)
                
                
                if total_saved >= target_total:
                    break
        
        print(f"{split.capitalize()} set: Saved {total_saved} images ({max_per_class} per class)")
        return total_saved

    train_data = CIFAR10(root='data/raw', train=True, download=True)
    test_data = CIFAR10(root='data/raw', train=False, download=True)
    
    train_saved = save_split(train_data, "train", max_per_class_train)
    test_saved = save_split(test_data, "test", max_per_class_test)
    


if __name__ == "__main__":
    downsample_and_prepare_cifar10()