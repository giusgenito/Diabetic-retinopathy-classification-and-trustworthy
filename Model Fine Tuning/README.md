# Fine-Tuning DINOv2 for Diabetic Retinopathy Classification

This repository contains the code and methodology for fine-tuning the DINOv2 (ViT-G/14) model for the task of Diabetic Retinopathy (DR) severity classification. The project leverages a powerful pre-trained vision transformer and adapts it using a strategic fine-tuning approach on a composition of several public DR datasets.

The notebook handles the entire pipeline: from data loading and preprocessing to training, evaluation, and confidence calibration.

---

## 📊 Experimental Protocol

Leave One Domain Out experimental protocol uses a combination of several datasets for training and a separate dataset for testing. 
---

## ✨ Key Features

* **State-of-the-Art Model**: Utilizes **DINOv2 (ViT-G/14)**, a powerful self-supervised vision model from Meta AI, as the feature extraction backbone.
* **Strategic Fine-Tuning**: Employs a partial fine-tuning strategy, unfreezing only the last 5 blocks of the transformer and a custom classification head. This balances feature adaptation with computational efficiency and helps prevent catastrophic forgetting.
* **Multi-Dataset Training**: Aggregates several DR datasets (DeepDRiD, RLDR, Messidor-2, FGADR, IDRiD) to create a robust and diverse training set.
* **Comprehensive Training Pipeline**: Includes features like:
    * **Checkpointing**: Saves the model state at every epoch, allowing training to be resumed.
    * **Early Stopping**: Monitors validation loss and stops training if no improvement is seen after a defined `patience` period, preventing overfitting.
    * **AdamW Optimizer**: Uses an advanced optimization algorithm with different learning rates for the backbone and the classification head.
* **Robust Evaluation**: Calculates a wide range of metrics on the test set, including Accuracy, F1 Macro score, Area Under the Curve (AUC), and Cohen's Kappa (with linear weighting).
* **Confidence Calibration**: Implements **Temperature Scaling** as a post-processing step to calibrate the model's output probabilities, making them more reliable.

---

## 🔧 Model Architecture & Fine-Tuning

The core of this project is the **DINOv2 Vision Transformer (ViT-G/14)**. Instead of training the entire network from scratch, we use a more efficient fine-tuning approach:

1.  **Backbone Freezing**: All parameters of the DINOv2 backbone are initially frozen (`param.requires_grad = False`).
2.  **Partial Unfreezing**: To adapt the high-level features to the specific domain of retinal images, the parameters of the **last 5 transformer blocks** (blocks 35 to 39) are unfrozen, allowing them to be updated during training.
3.  **Classification Head**: A custom classification head is added on top of the backbone. It takes the `1536-dimensional` feature vector from DINOv2 and passes it through a sequence of layers:
    * Linear layer (1536 -> 512)
    * ReLU activation
    * Dropout (p=0.3)
    * Final Linear layer (512 -> 5 classes)

This strategy allows the model to learn domain-specific features while retaining the powerful, general-purpose knowledge learned during DINOv2's self-supervised pre-training.

---

## 🚀 How to Run the Project

### 1. Prerequisites

Ensure you have the necessary Python libraries installed. You can install them using pip:

```bash
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

Da completare
```

2. Dataset Setup
Before running, you must configure the dataset paths in the second code cell of the notebook.
```python
# Main paths
train_data_paths = [
   "/path/to/your/3_DeepDRiD",
   "/path/to/your/6_RLDR",
   # ... and other training datasets
]

test_data_path = "/path/to/your/2_APTOS"
```

The script expects the data to be organized in class-based folders (e.g., .../3_DeepDRiD/0, .../3_DeepDRiD/1, etc.).
3. Training
Execute the cells sequentially. The main training function is train_model. When you run the final cell, you will be prompted to decide whether to resume training from a checkpoint:
```bash
Vuoi continuare l'addestramento o ripartire? (y/Yes/yes per continuare)
```
- Enter y or yes to load the latest checkpoint from the /home/jupyter-sdm/GENITO/LAVORO_COMPLETO/Checkpoint_training/ directory and continue training.
- Enter anything else to start training from scratch.

The training process will display a progress bar for each epoch and print the validation metrics. The best-performing model (based on validation loss) will be saved as best_dinov2_model_con_preprocessing.pth.

4. Evaluation
After the training is complete, the subsequent cells will:

- Load the best_dinov2_model_con_preprocessing.pth model.

- Run inference on the test set (2_APTOS dataset in this configuration).

- Calculate and display the final performance metrics (Accuracy, F1, AUC, Kappa).

- Generate and show a bar chart of the metrics and a detailed confusion matrix.

5. Confidence Calibration
The final cell demonstrates how to use Temperature Scaling to calibrate the model's confidence scores. It loads the best model, finds the optimal temperature using the validation set, and reports the Expected Calibration Error (ECE) before and after scaling.
