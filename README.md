# **Image Classification: Male vs. Female Using CNN in TensorFlow**

This repository contains an implementation of an image classification model using Convolutional Neural Networks (CNNs) in TensorFlow. The model is specifically designed to classify images into two categories: **Male** and **Female** based on visual characteristics.

## **Project Structure**

- **Data Preparation**: Download and organize the dataset.
- **Model Building**: Define a CNN model to extract features distinguishing male and female images.
- **Training and Evaluation**: Train and validate the model, using data augmentation to improve performance.
- **Prediction**: Apply the model to new images to classify them as either Male or Female.

---

## **Table of Contents**

1. [Setup](#setup)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Prediction](#prediction)
7. [Results and Plots](#results-and-plots)
8. [Conclusion](#conclusion)

---

## **1. Setup**

Clone this repository and install the necessary dependencies.

```bash
git clone https://github.com/your-repo/image-classification-cnn.git
cd image-classification-cnn
pip install -r requirements.txt
```

## **2. Dataset**

The dataset consists of images organized into two folders by class labels: `Laki-Laki` (Male) and `Perempuan` (Female). We use a 60/40 split for training and validation. For each class, sample images are displayed to verify correct data structure.

## **3. Model Architecture**

The CNN model follows a classic design:

- **Convolution Layers**: Extract spatial features that help differentiate between male and female images.
- **Max Pooling**: Reduce dimensionality while retaining critical features.
- **Flattening and Dense Layers**: Transform features into a vector and apply fully connected layers.
- **Output Layer**: Uses a sigmoid activation function for binary classification (Male vs. Female).

## **4. Training**

The model is trained on the dataset with a binary cross-entropy loss function and RMSprop optimizer. Data augmentation (random flips, rotations, and zooms) is applied to enhance model robustness and generalization.

## **5. Evaluation**

Accuracy and loss are tracked for both training and validation sets over multiple epochs. Plots of these metrics help analyze model performance and detect any overfitting.

## **6. Prediction**

The model can predict new images, displaying the classification result (Male or Female) and confidence score.

## **7. Results and Plots**

### Training and Validation Accuracy and Loss

The model’s performance is visualized with plots showing the training and validation accuracy and loss. These plots offer insights into model convergence and the effectiveness of data augmentation.

---

## **8. Conclusion**

This project demonstrates image classification to distinguish between Male and Female images. Key features, such as data augmentation and an optimized dataset pipeline, enhance model accuracy and robustness, making it adaptable to new, unseen images.

---

## **License**

This project is open-source and available for use under the MIT License.
