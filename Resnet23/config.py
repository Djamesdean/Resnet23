# config.py

# Path to processed dataset (after downsampling)
DATA_DIR = "data/processed"

# CIFAR-10 class names
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# Target image size (CIFAR-10 is already 32x32, but useful for resizing if needed)
IMAGE_SIZE = (32, 32)

# Model configuration (example — useful for training/eval scripts later)
NUM_CLASSES = 10
BATCH_SIZE = 64
RANDOM_SEED = 42

# Paths to store model checkpoints, logs, etc. (optional)
MODEL_DIR = "models"
LOG_DIR = "reports/logs"