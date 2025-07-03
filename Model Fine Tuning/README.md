# Retinal Image Classification with DINOv2

This repository implements a deep learning pipeline for retinal image classification using a DINOv2 model in PyTorch. The code includes dataset preparation, model training with validation, and detailed evaluation using various performance metrics and visualizations.

## Features

- **Easy dataset preparation:** Load retinal images and apply data augmentations and transformations.
- **Flexible training loop:** Includes model checkpointing, early stopping, and the option to continue interrupted training.
- **Comprehensive evaluation:** Computes accuracy, F1-score, AUC, Cohen’s kappa, with confusion matrix and results visualization.
- **Highly customizable:** Modular code, easy to adapt for other datasets or models.

## Installation Requirements

To run this project, ensure you have Python 3.8+ and install the following packages.  
Some libraries require installing from specific indexes for CUDA support.

**Create a `requirements.txt` file:**
```
--extra-index-url https://download.pytorch.org/whl/cu117
torch==2.0.0
torchvision==0.15.0
omegaconf
torchmetrics==0.10.3
fvcore
iopath
xformers==0.0.18
submitit
--extra-index-url https://pypi.nvidia.com
cuml-cu11
Pillow
scikit-learn
matplotlib
tqdm
numpy
seaborn
opencv-python
pytorch-grad-cam
```

**Installation:**
```bash
python -m venv venv
source venv/bin/activate             # On Windows: venv\Scripts\activate
pip install --extra-index-url https://download.pytorch.org/whl/cu117 -r requirements.txt
pip install --extra-index-url https://pypi.nvidia.com cuml-cu11
```
*Note: `cuml-cu11` is best installed separately as shown.*

## Usage

Run the training script with:
```bash
python DinoV2.py
```
If a previous checkpoint exists, you’ll be prompted to continue training or start anew.

## Dataset Structure

Make sure your image dataset directories (as set inside `DinoV2.py`) follow this structure:
```
dataset_root/
  train/
    0/
    1/
    ...
  val/
    0/
    1/
    ...
  test/
    0/
    1/
    ...
```
Each class should be placed in its own subfolder (e.g. `0`, `1`, etc.).

## Visualization

After training, the script displays learning curves and confusion matrices for model diagnosis and interpretation.
