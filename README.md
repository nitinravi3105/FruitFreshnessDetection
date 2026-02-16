# Fruit & Vegetable Freshness Detection System

## Overview
This project implements a Deep Learning-based Computer Vision system for detecting the freshness of fruits and vegetables using Transfer Learning. The model classifies input images into 18 categories representing fresh and rotten classes.

## Features
- Multi-class image classification (18 classes)
- Transfer Learning using MobileNetV2
- Image upload-based prediction
- Real-time webcam inference
- Prediction confidence scoring
- Probability distribution visualization

## Technologies Used
- Python
- TensorFlow / Keras
- Streamlit
- OpenCV
- NumPy
- Pillow

## Dataset
The dataset consists of images representing fresh and rotten instances of:

- Apple
- Banana
- Bitter Gourd
- Capsicum
- Cucumber
- Okra
- Orange
- Potato
- Tomato

## Model
The system utilizes MobileNetV2 with Transfer Learning for efficient feature extraction and accurate classification. Data preprocessing and augmentation techniques were applied to improve model generalization.

## How to Run

Install dependencies:

```bash
pip install tensorflow streamlit opencv-python pillow numpy
