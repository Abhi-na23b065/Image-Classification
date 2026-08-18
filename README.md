# Shift-Guard 10 – Robust Image Classification

## 📌 Project Overview

This project implements an image classification pipeline for the **Shift-Guard 10 Robust Image Classification Challenge**.

The model is a custom **Convolutional Neural Network (CNN)** implemented using **PyTorch**. It uses residual blocks, data augmentation, label smoothing, SGD optimization, learning-rate warmup, cosine decay, and test-time augmentation (TTA) to improve classification performance.

The notebook trains the model on the provided training images, evaluates it using **Macro F1-score**, saves the best-performing model, and generates predictions for the test set.

---

## 🎯 Objective

The objective is to classify input images into one of **10 classes**.

The primary evaluation metric used in the notebook is:

**Macro F1-score**

Macro F1 gives equal importance to every class and is therefore useful when evaluating performance across multiple classes.

---

## 🗂️ Dataset Structure

The notebook expects the dataset at:

```text
/kaggle/input/competitions/shift-guard-10-robust-image-classification-challenge/
```

The following files and directories are used:

```text
shift-guard-10-robust-image-classification-challenge/
│
├── train_images/
│   └── training images
│
├── test_images/
│   └── test images
│
├── train_labels.csv
├── sample_submission.csv
└── classes.txt
```

`classes.txt` contains the class names used by the classifier.

---

## ⚙️ Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Pandas
* Scikit-learn
* PIL
* Matplotlib

---

## 🔄 Project Pipeline

The overall workflow is:

```text
Dataset
   ↓
Load class names and labels
   ↓
Train / Validation Split
   ↓
Image Dataset & DataLoader
   ↓
Data Augmentation
   ↓
Custom CNN with Residual Blocks
   ↓
Model Training
   ↓
Validation using Macro F1
   ↓
Save Best Model
   ↓
Test-Time Augmentation
   ↓
Generate Predictions
   ↓
submission.csv
```

---

## 📊 Data Preparation

The dataset is divided into:

* **90% training data**
* **10% validation data**

A stratified split is used so that the class distribution is maintained between training and validation sets.

The random seed is set to `42` for reproducibility.

---

## 🖼️ Data Augmentation

The training pipeline applies several augmentations:

* Random crop
* Random horizontal flip
* Color jitter
* Random erasing
* Normalization

Training images are normalized using:

```python
MEAN = [0.4914, 0.4822, 0.4465]
STD  = [0.2470, 0.2435, 0.2616]
```

Validation images only undergo tensor conversion and normalization.

---

## 🧠 Model Architecture

The project uses a custom CNN called:

```text
StrongCNN
```

The network contains:

### Entry Layer

```text
3-channel RGB image
        ↓
3 × 3 Convolution
        ↓
Batch Normalization
        ↓
ReLU
```

### Residual Layers

The model contains three main residual stages:

```text
Layer 1: 64 → 64
Layer 2: 64 → 128
Layer 3: 128 → 256
```

The second and third layers downsample the feature maps using stride 2.

Each residual block contains:

* 3×3 convolution
* Batch normalization
* ReLU
* 3×3 convolution
* Batch normalization
* Shortcut connection

### Classification Head

The final feature maps are processed using:

```text
Adaptive Average Pooling
        ↓
Flatten
        ↓
Dropout (0.3)
        ↓
Linear Layer
        ↓
10 class predictions
```

---

## 🏋️ Training Configuration

The model is trained for:

```text
200 epochs
```

with a batch size of:

```text
256
```

The optimizer is **SGD** with:

```text
Learning rate: 0.1
Momentum: 0.9
Weight decay: 5e-4
Nesterov: True
```

The loss function is:

```python
CrossEntropyLoss(label_smoothing=0.05)
```

---

## 📉 Learning Rate Schedule

The notebook uses a two-stage learning-rate schedule:

### Warmup

The learning rate is gradually increased during the first **5 epochs**.

### Cosine Decay

After warmup, the learning rate follows a cosine-decay schedule until the end of training.

This allows the model to start training more gradually and then reduce the learning rate as training progresses.

---

## 💾 Model Checkpointing

After every epoch, the model is evaluated on the validation set.

If the validation Macro F1 improves, the model weights are saved:

```text
best_model.pth
```

Only the best-performing model according to validation Macro F1 is retained.

---

## 📈 Evaluation

The notebook evaluates the model using:

* Training loss
* Training accuracy
* Validation loss
* Validation Macro F1-score
* Per-class precision
* Per-class recall
* Per-class F1-score

A classification report is generated at the end of the notebook.

---

## 🔍 Test-Time Augmentation

For test prediction, the notebook uses **5 TTA rounds**.

Each round applies:

* Random crop
* Random horizontal flip
* Normalization

The model produces class probabilities for each augmented version.

The probabilities from all five rounds are averaged:

```text
TTA 1
TTA 2
TTA 3
TTA 4
TTA 5
 ↓
Average probabilities
 ↓
Select class with highest probability
```

This produces the final test prediction.

---

## 📄 Output

The final predictions are saved as:

```text
submission.csv
```

The saved submission follows the structure of the provided `sample_submission.csv`.

---

## 🚀 How to Run

### 1. Open the notebook

The project is designed around a Kaggle dataset path.

### 2. Make sure the dataset is available

Ensure the following files exist:

```text
train_images/
test_images/
train_labels.csv
sample_submission.csv
classes.txt
```

### 3. Run the notebook

Execute the cells in order.

The notebook will:

1. Load the dataset
2. Create the train/validation split
3. Build the CNN
4. Train for 200 epochs
5. Save the best model
6. Generate test predictions
7. Create `submission.csv`
8. Generate a per-class F1 report

---

## 📁 Important Output Files

| File             | Purpose                                         |
| ---------------- | ----------------------------------------------- |
| `best_model.pth` | Best model weights based on validation Macro F1 |
| `submission.csv` | Final test predictions                          |

---

## 📝 Notes

* The notebook automatically uses CUDA when a GPU is available.
* Otherwise, it falls back to CPU.
* The random seed is fixed to `42`.
* The model uses residual connections to improve feature learning.
* The validation set uses stratified sampling.
* The final prediction uses 5-round test-time augmentation.

---

## 👨‍💻 Project Summary

This project demonstrates an end-to-end deep-learning image classification pipeline using PyTorch. It combines a custom residual CNN architecture with data augmentation, regularization, learning-rate scheduling, checkpointing, and test-time augmentation to build a robust image classifier for the Shift-Guard 10 classification challenge.
