# Employee Safety Gear Detection Using Deep Learning (MobileNetV2)

## Introduction

Ensuring that employees wear the appropriate safety gear (such as helmets and safety vests) is crucial in industrial and construction environments to prevent accidents and ensure compliance with safety protocols. This project uses **MobileNetV2**, a state-of-the-art deep learning model, to detect whether employees are wearing safety gear or not, based on image inputs.

The dataset consists of two categories of images:
- **Wearing Safety Gear**: Images of individuals wearing helmets and safety vests.
- **Not Wearing Safety Gear**: Images of individuals without any safety gear.

This model can be deployed for real-time monitoring in industrial settings to ensure safety compliance before employees begin their tasks.

---

## Dataset

The dataset used for this project is organized into two subfolders:
1. **Wearing Safety Gear**: Contains images of individuals with safety equipment.
2. **Not Wearing Safety Gear**: Contains images of individuals without safety gear.

The data is split into training, validation, and test sets, and the images are augmented using Keras' `ImageDataGenerator` for better generalization.

---

## Model Architecture

This project uses **MobileNetV2**, a lightweight convolutional neural network architecture designed for mobile and embedded vision applications. The key components of the model include:

- **Depthwise Separable Convolutions** for computational efficiency.
- **Inverted Residual Blocks** to enhance performance while reducing the number of parameters.
- **Global Average Pooling** to reduce feature map dimensions before classification.
- **Binary Classification**: The model classifies images into two categories — "Wearing Safety Gear" and "Not Wearing Safety Gear."

The model was trained on an augmented dataset, and the weights are fine-tuned using transfer learning from the pre-trained MobileNetV2 model on the ImageNet dataset.

---

## Installation and Setup

### Prerequisites

Ensure you have the following installed:
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- OpenCV
- Scikit-learn
- ImageDataGenerator (from Keras)

## Prepare the dataset:
- Organize your dataset into two subfolders (Wearing Safety Gear and Not Wearing Safety Gear) and place them under a directory, e.g., dataset/.
- Ensure your folder structure looks like this
dataset/
  Wearing Safety Gear/
  Not Wearing Safety Gear/

## How to Use
- Training the Model
You can train the model by running the cell model . This cell reads the dataset, augments the images, and trains the MobileNetV2 model. It will save the trained model as your_model.h5

- Testing the Model
Use the test cell to evaluate the model on the test set. The cell will load the saved model (your_model.h5), make predictions on the test images, and display random images with the model's predictions.

## Example Output
For each test image, the model will display whether the prediction is correct or incorrect.
Image 1:
True Label: Wearing Safety Gear
Predicted Label: Wearing Safety Gear
Correct Prediction!

Image 2:
True Label: Not Wearing Safety Gear
Predicted Label: Wearing Safety Gear
Incorrect Prediction.

## Results
The MobileNetV2 model was trained on the augmented dataset, and the following results were observed on the test set:
- Accuracy: 96%
- AUC: 0.99 
The model can efficiently classify whether an employee is wearing safety gear based on real-time images.

## Model Summary

Here is a brief summary of the MobileNetV2 model used in this project:

![Alt text](https://raw.github.com/FatimaaAlzahraa/Employee-Safety-Gear-Detection-Using-Deep-Learning/blob/master/model.png)


- Base Model: MobileNetV2 (pre-trained on ImageNet)
- Global Average Pooling: Reduces the spatial dimensions of the feature maps.
- Dense Layers: Two dense layers are added on top of the base model. The first dense layer has 128 units with a sigmoid activation, followed by a single output unit for binary classification.
The model is highly efficient due to the lightweight nature of MobileNetV2, making it suitable for real-time safety compliance monitoring.

## Streamlit Deployment

You can try out the **Employee Safety Gear Detection** app live on Streamlit Cloud:

- **[Click here to view the live app](https://safety-gear-detections.streamlit.app/)**



