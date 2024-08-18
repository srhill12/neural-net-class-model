
# Neural Network Binary Classification Model

This project demonstrates the use of a Keras Sequential model to classify data points based on two features. The model is trained on a dataset of 5,000 observations and evaluated for its performance in predicting binary outcomes.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Overview](#data-overview)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Training History](#training-history)
- [Analysis and Inference](#analysis-and-inference)

## Overview

This project involves building a neural network using TensorFlow and Keras to perform binary classification on a dataset. The data contains two features (`Feature 1` and `Feature 2`) and a binary target variable (`Target`). The goal is to train the model to accurately classify the data points into two categories.

## Installation

To run this project, you need to have Python installed along with the necessary libraries. You can install the required dependencies by running:

```bash
pip install -r requirements.txt
```

## Project Structure

The project files are organized as follows:

```plaintext
├── nn_classification.ipynb         # Jupyter notebook with data analysis and model training
├── README.md                       # Project documentation
├── requirements.txt                # List of required libraries
```

## Data Overview

The dataset used in this project contains 5,000 observations with two features (`Feature 1`, `Feature 2`) and a binary target variable (`Target`):

```plaintext
Feature 1  | Feature 2  | Target
-----------|------------|-------
0.523      | -0.678     | 0
1.234      | 1.567      | 1
...        | ...        | ...
```

## Data Preprocessing

Before training the model, the data is preprocessed by scaling the features. This step ensures that the features are on a similar scale, which improves the model's convergence during training.

### Steps:

1. **Data Splitting**: The dataset is split into training and testing sets using `train_test_split`.
2. **Feature Scaling**: A `StandardScaler` is used to scale the features in both the training and testing sets.

Here’s the code snippet for preprocessing:

```python
X_scaler = StandardScaler()
X_scaler.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
```

## Model Architecture

The neural network model is built using Keras' Sequential API. The model has the following architecture:

- **Input Layer**: Takes in the two features (`Feature 1` and `Feature 2`).
- **Hidden Layer**: A Dense layer with 5 units and ReLU activation function.
- **Output Layer**: A Dense layer with 1 unit and sigmoid activation function, providing a probability for the binary classification.

Here’s the code snippet for defining the model:

```python
nn_model = tf.keras.models.Sequential()
nn_model.add(tf.keras.layers.Dense(units=5, activation="relu", input_dim=input_nodes))
nn_model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
```

## Model Training

The model is compiled using the binary cross-entropy loss function and the Adam optimizer. The model is then trained for 50 epochs using the scaled training data.

### Training Code:

```python
nn_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
fit_model = nn_model.fit(X_train_scaled, y_train, epochs=50)
```

## Evaluation

The model's performance is evaluated on the test set, with the loss and accuracy being the primary metrics.

### Evaluation Code:

```python
model_loss, model_accuracy = nn_model.evaluate(X_test_scaled, y_test, verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
```

The final evaluation on the test set resulted in the following metrics:

```plaintext
Loss: 0.12407264113426208
Accuracy: 0.9527999758720398
```

## Training History

The training history was recorded and can be visualized by plotting the accuracy over the epochs. This provides insight into the model's learning process.

### Training History Plot:

```python
history_df = pd.DataFrame(fit_model.history, index=range(1, len(fit_model.history["loss"]) + 1))
history_df.plot(y="accuracy")
```

## Analysis and Inference

### Analysis

The neural network model achieved an accuracy of approximately 95% on the test dataset. The following factors contribute to this performance:

1. **Model Simplicity**: The model has a simple architecture with one hidden layer, which is well-suited for this dataset.
2. **Feature Scaling**: The use of feature scaling ensures that the model converges efficiently during training.
3. **Training Stability**: The loss function shows a stable decline over the epochs, indicating that the model is learning effectively.

### Inference

Given the high accuracy score, the following inferences can be drawn:

- **High Performance**: The model performs well on the given dataset, achieving a high accuracy of around 95%.
- **Potential for Generalization**: The stable training and test accuracy suggest that the model is likely generalizing well, though further testing on different datasets would be advisable.
- **Model Validation**: While the accuracy is high, it's important to consider additional validation techniques, such as cross-validation, to ensure the model's robustness and reliability in real-world applications.

Overall, the neural network model provides a solid performance on the binary classification task, demonstrating the effectiveness of deep learning approaches for this type of problem.
