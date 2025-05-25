# config.py


DATA_DIR = "data/processed"


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


IMAGE_SIZE = (32, 32)


NUM_CLASSES = 10
BATCH_SIZE = 64
RANDOM_SEED = 42


MODEL_DIR = "models"
LOG_DIR = "reports/logs"