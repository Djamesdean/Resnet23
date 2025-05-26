
# 🧠 ResNet23 CIFAR-10 Classifier

This project implements an **image classification pipeline** using a **custom ResNet23 model** with 23 layers starting from the original resnet18 on a **downsampled CIFAR-10 dataset**.

---

## 📁 Project Structure

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         Resnet23 and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── Resnet23   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes Resnet23 a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

---

## Setup

### Create Conda Environment

```bash
conda create -n Resnet23 python=3.10
conda activate Resnet23
```

### Install Requirements

```bash
pip install torch torchvision tqdm
```

---

## Configuration

Defined in `Resnet23/config.py`:

```python
DATA_DIR = "data/processed"
IMAGE_SIZE = (32, 32)
BATCH_SIZE = 64
NUM_CLASSES = 10
CLASS_NAMES = ["airplane", "automobile", ..., "truck"]
```

This helps keep training code clean and centralized.

---

## Download & Downsample CIFAR-10

Run this script to download CIFAR-10 and save a **limited number of images per class** into a folder-based structure:
- Downloads the dataset into data/raw.
- Creates target directories if they don't exist.
- Iterates over (image, label) pairs.
- Converts image from tensor to PIL using ToPILImage.
- Resizes image to IMAGE_SIZE (usually 32×32).
- Saves image to output_dir/split/class_name/filename.png.

### Arguments :
- **output_dir (str)**: Destination directory for processed data .
- **max_per_class (int)**: Maximum number of images to save for each class (default is 500).


```bash
python Resnet23/download_data.py
```

This creates:

```
data/processed/
  ├── train/
  │   ├── airplane/
  │   ├── ...
  └── test/
      ├── airplane/
      ├── ...
```

---

## 🧾 4. Dataset Loader

Implemented in `Resnet23/dataset.py`:

- `CIFARCustomDataset`: Loads images from folder structure.
- `get_default_transform()`: Defines transforms (resize, normalize).

### Parameters:
- **root_dir (str)**: Path to the train/ or test/ directory containing class-named folders.
- **transform**: A set of transforms to apply to each image (e.g., resizing, normalization).

### Process:
- Iterates over all class folders.
- Maps each .png image to its label index.
- Stores:
  - **self.image_paths**: List of full paths to all images.
  - **self.labels**: Corresponding label indices.
  
- __getitem__ : 
    - Loads image path from self.image_paths.
    - Reads the image using PIL.Image.open().
    - Converts it to RGB.
    - Applies any provided transform.
    - Returns the image tensor and its label.


---

## Model Architecture

Defined in `Resnet23/modeling/model23.py`:

This file contains a custom implementation of a 23-layer ResNet (ResNet-23) based on the original ResNet architecture. The model extends the standard ResNet-18 design with additional layers and modern training enhancements to improve performance on our task.

### Model Specifications
- **Total Layers:** 23 convolutional layers
- **Block Type:** BasicBlock (2 conv layers per block)
- **Layer Configuration:** [3, 3, 3, 2] blocks per layer
- **Parameters:** ~11.7M (compared to ~11.2M in ResNet-18)
- **input Size:**  32×32 for CIFAR-10
### Architectural Details :
#### 1. Layer Count 

- **ResNet-18**: [2, 2, 2, 2] blocks = 18 layers
- **ResNet-23**: [3, 3, 3, 2] blocks = 23 layers
- **Impact**: Increased model  and  power

#### 2. Input for Small Images

- **Traditional**: 7×7 conv, stride=2, MaxPool → Reduces 224×224 to 56×56
- **ResNet-23**: 3×3 conv, stride=1, no MaxPool → Preserves 32×32 resolution
- **Benefit** : Prevents  downsampling of small images

#### 3. Dropout

- **Block Dropout**: 2D dropout (default 10%) within residual blocks
- **Final Dropout**: Additional 20% dropout before classification
- **Benifit**: Prevents overfitting on small datasets
#### 4. Make Layer method
- This method creates a sequence of residual blocks (BasicBlock) that form one layer of the ResNet.
  - Checks if downsampling is needed (if stride != 1 or channels change).If yes, creates a shortcut (downsample).
  - Adds the first block (handles the dimension change).
  - Adds more blocks (keeps dimensions the same).
  - Returns the full stack as a nn.Sequential layer.
```
Input (64 channels, 32x32)
│
▼
[Block 1] → Changes to 128 channels, size 16x16 (stride=2)
│
▼
[Block 2] → Keeps 128 channels, 16x16
│
▼
[Block 3] → Keeps 128 channels, 16x16
│
▼
Output (128 channels, 16x16)
```
#### 5. Advanced Weight 
Initializes the neural network weights for better training performance.
```python
def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
```
-  Kaiming Normal Initialization (best for ReLU in Conv layers)
-  BatchNorm Initialization (sets weights=1, biases=0)
-  Linear Layer Initialization (small random weights for stability)

---

## 🏋️‍♂️ 6. Training Script

Located in `Resnet23/train.py`:

- Loads data using `DataLoader`
- Trains model using `CrossEntropyLoss` and `Adam`
- Tracks train/test accuracy
- Saves model to `models/resnet23_cifar.pth`

Run training:

```bash
python Resnet23/train.py
```


---

##  Results 
 Training the model for 20 epochs with total of 5000 images gave us : 
 - **50% Train accuracy / 40% Test accuracy** --> Using Typical Resnet18. 
 - **91% Train accuracy / 58% Test Accuracy** --> Using modified Resnet23.

---
## Discussion
The main reason why the model gave us a high **training accuracy** and lower **test accuracy** is that the model was too powerful for our small dataset so it memorized our data which lead us to have a significant difference betweeen train and test .
- **Solution** : train it on more data will resolve the issue .

