# Brain Tumor Classification Project

## Introduction:
This project focuses on classifying brain MRI scans into tumor-positive and tumor-negative categories. 
Brain tumors are critical to detect early, and MRI imaging provides a non-invasive method to observe 
abnormal tissue. Due to the limited size of medical imaging datasets, deep learning models often require
careful training strategies to achieve reliable performance.

## Data Handling:
The dataset is organized into 'yes' (tumor) and 'no' (healthy) folders. Custom PyTorch Dataset and 
LightningDataModule classes are implemented to efficiently load and preprocess images. 
Transformations applied include resizing, cropping, rotation, grayscale conversion to 3 channels, 
and normalization to standardize pixel values. Data is split into training, validation, and test sets 
with stratification to maintain class balance.

## Methods:
1. Baseline CNN:
   - A custom convolutional neural network (PyTorchCNN) with multiple convolutional, batch normalization, 
     ReLU, pooling, and dropout layers.
   - Fully connected layers at the end for classification.
   - Trained from scratch to provide a baseline performance.

2. Transfer Learning with ResNet:
   - ResNet50 pretrained on ImageNet is used as a feature extractor.
   - Stage 1: Freeze all convolutional layers and train only a new classification head to adapt to our dataset.
   - Stage 2: Fine-tune the last convolutional block to improve feature extraction specific to brain MRI scans.
   

## Training and Evaluation:
- PyTorch Lightning is used to structure training, validation, and testing loops, including GPU acceleration.
- Accuracy, confusion matrix, and loss curves are logged using CSVLogger for monitoring.
- Baseline accuracy is calculated by predicting the majority class.
- Fine-tuning improved test accuracy significantly, achieving over 92% accuracy.
- Early stopping and model checkpointing are used to save the best model.

## Conclusion:
The project demonstrates how transfer learning can overcome limitations of small datasets in medical imaging.
Using pretrained ResNet50 with staged fine-tuning provides high classification accuracy compared to a 
baseline CNN trained from scratch. This pipeline is scalable and can be extended to other medical 
image classification tasks.


