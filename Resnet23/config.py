# config.py
import os

import mlflow

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



# === DagsHub/MLflow Configuration ===

DAGSHUB_USER = "Djamesdean"  
DAGSHUB_REPO = "Resnet23"
DAGSHUB_TOKEN = "ae1edde218d06eb411faecccc02be1210ee9c127"
TRACKING_URI = f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow"

def setup_experiment():
    try:
        os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER
        os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment("resnet23-cifar10")
        print("DagsHub connection successful")
        return True
    except:
        print("DagsHub connection failed")
        return False