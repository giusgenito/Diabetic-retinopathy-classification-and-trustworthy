# Fine-Tuning DINOv2 for Diabetic Retinopathy Classification

This repository contains the code and methodology for fine-tuning the DINOv2 (ViT-G/14) model for the task of Diabetic Retinopathy (DR) severity classification. The project leverages a powerful pre-trained vision transformer and adapts it using a strategic fine-tuning approach on a composition of several public DR datasets.

The notebook handles the entire pipeline: from data loading and preprocessing to training, evaluation, and confidence calibration.

---

## 📊 Dataset Distribution

The experimental protocol uses a combination of several datasets for training and a separate dataset for testing. The distribution of images across the different severity classes (0 to 4) for the training, validation, and test sets is visualized below. This ensures a clear understanding of the data balance.

![Distribuzione Istanze A Seguito Del Protocollo Sperimentale Applicato Ad APTOS](https://i.imgur.com/8Q0NlG9.png)

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
pip install torch torchvision
pip install scikit-learn matplotlib seaborn tqdm opencv-python
pip install pytorch-grad-cam
pip install temperature-scaling
