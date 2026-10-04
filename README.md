# Advanced Machine Learning Tasks

**Author:** Islambek Ziyash

## Overview

Two advanced machine learning assignments covering dimensionality reduction, classical classifiers on image data, and deep learning with CNNs.

## Project Structure

```
du1/
   ├── homework_01_B242.ipynb   # Dimensionality reduction + binary classification
   ├── train.csv                # Training data (28×28 grayscale images, flattened)
   ├── evaluate.csv             # Unlabelled evaluation data
   └── submission.csv           # Predicted labels (submission)
du2/
    ├── homework_02_B242.ipynb   # Neural network multi-class classification
    ├── train.csv                # Training data (32×32 grayscale images, flattened)
    ├── evaluate.csv             # Unlabelled evaluation data
    ├── best_cnn_model.keras     # Saved best CNN model
    ├── best_cnn_model.h5        # Saved best CNN model (HDF5 format)
    └── results.csv              # Predicted labels (submission)
```

---

## Homework 1 — Dimensionality Reduction & Classification

**Task:** Binary classification on high-dimensional image data (28×28 grayscale images → 784 features). The challenge is to handle the high dimensionality effectively before applying classifiers.

### Preprocessing
- Pixel values normalized to `[0, 1]` by dividing by 255
- Stratified 70 / 15 / 15 train / validation / test split
- PCA applied across multiple component counts (10, 20, 50, 100, 150) to benchmark the effect of dimensionality on accuracy

### Models Compared
| Model | Notes |
|---|---|
| SVM (RBF kernel) | `GridSearchCV` over `C` ∈ {1, 10}, `gamma` ∈ {0.01, 0.001} |
| SVM (Linear kernel) | `GridSearchCV` over `C` ∈ {0.1, 1, 10} |
| Gaussian Naive Bayes | Baseline; also used for synthetic image generation |
| Linear Discriminant Analysis (LDA) | Also used for synthetic image generation |

All models tested with 3-fold cross-validation, optimising **accuracy**.

### Dimensionality Reduction
- **PCA** applied with 10–150 components; accuracy of all four models compared across each dimension
- **SVM (RBF)** with 50 PCA components selected as the final model based on the accuracy/speed trade-off
- Final pipeline: `PCA(n_components=50) → SVC(kernel='rbf', C=1)`

### Generative Analysis
- **Gaussian NB**: synthetic images generated per class using class-conditional mean and diagonal covariance
- **LDA**: synthetic images generated using class-conditional means and shared covariance matrix

### Evaluation
- Accuracy, confusion matrix, and classification report on validation set
- Best model applied to `evaluate.csv` to produce `submission.csv`

---

## Homework 2 — Neural Networks for Image Classification

**Task:** Multi-class classification on 32×32 grayscale images using neural networks.

### Preprocessing
- Pixel values normalized to `[0, 1]`
- Labels one-hot encoded with `to_categorical`
- Images reshaped to `(32, 32, 1)` for CNN input
- Stratified 70 / 15 / 15 train / validation / test split

### Models

#### Baseline MLP
```
Dense(512, relu) → Dropout(0.3) → Dense(256, relu) → Dropout(0.3) → Dense(N, softmax)
```
- Optimizer: Adam · Loss: categorical crossentropy
- Result: ~87.0% train accuracy, ~85.1% validation accuracy

#### MLP with GridSearchCV Tuning
- Tuned via `KerasClassifier` + `GridSearchCV`
- Parameters: `units` ∈ {256, 512}, `dropout` ∈ {0.2, 0.3, 0.4}, `optimizer` ∈ {adam, rmsprop}

#### CNN (Final Model)
```
Conv2D(32, 3×3, relu) → MaxPooling2D(2×2)
→ Conv2D(64, 3×3, relu) → MaxPooling2D(2×2)
→ Flatten → Dense(128, relu) → Dropout(0.3) → Dense(N, softmax)
```
- Optimizer: Adam · Loss: categorical crossentropy · Epochs: 30

#### CNN with BatchNormalization + Tuning
- Extended CNN with `BatchNormalization` after each Conv and Dense block
- Tuned with `GridSearchCV` over `dropout_rate`, `filters_1`, `filters_2`, `dense_units`
- Best model saved as `best_cnn_model.keras` and `best_cnn_model.h5`

### Evaluation
- Training/validation accuracy and loss curves plotted per epoch
- Misclassified test samples visualised
- Final CNN model used to generate `results.csv` on evaluation data

---

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow scikeras
```

## Running

```bash
jupyter notebook "du1/homework_01_B242.ipynb"
jupyter notebook "du2/homework_02_B242.ipynb"
```

Both notebooks expect `train.csv` and `evaluate.csv` to be present in the same directory as the notebook. GPU is recommended for the CNN training in HW2.
