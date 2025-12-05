# CSCI4050 - Visisign: ASL Translation project

## Proposal/Project Specifications
[Link to proposal](https://github.com/Megadast/CSCI4050-machine-learning/blob/main/Documents/Project%20Proposal.pdf)

## Project Submission and details page
[Link to proposal](https://docs.google.com/document/d/1fHhpxRTeGKunU2NyJhxFJ5Jxy0uY3lJdPAncQGu8nMs/edit?tab=t.0)

## Table of Contents
[Table of Contents](#table-of-contents)   
[List of Figures](#list-of-figures)

- [1.0 Design Proposal](#10-design-proposal)
  - [1.1 Project Requirements and Specifications](#11-project-requirements-and-specifications)
- [2.0 Libraries Required](#20-libraries-required)
- [3.0 Dataset(s)](#30-datasets)
- [4.0 Integration](#40-integration)
  - [4.1 Phase One](#41-phase-one)
  - [4.2 Phase Two](#42-phase-two)
  - [4.3 Phase Three](#43-phase-three)
- [5.0 Firmware Code](#50-firmware-code)
  - [5.1 main.py](#51-mainpy)
- [6.0 Acronyms](#60-acronyms)
- [7.0 References](#70-references)

## List of Figures   
[Figure 1: Gantt Chart 2023](#figure-1-gantt-chart-2023)     

### 1.0 Design proposal
This project involves the development of a machine learning model capable of recognizing hand gestures connected to American Sign Language (ASL).
The scope is to recognize letters and/or digits from images and possibly expand to simple phrases like "Hello" or "Goodbye".
The system will take an image of a hand performing a sign as input and output the corresponding alphabet, number or simple word.
The idea is to use static images at first and try to expand into live translation.
<br>

<b>Design Approach (For now)</b>

```mermaid
flowchart LR
InputImage --> Preprocessing
Preprocessing --> HandCrop
HandCrop --> CNNModel
CNNModel --> PredictionOutput
PredictionOutput --> UserInterface
```
  
### 1.1 Project Requirements and Specifications   
- Implement automated dataset downloading using Kaggle API
- Preprocess images using PyTorch transforms
- Train a CNN classifier capable of recognizing ASL (American Sign Language)
- Build a prediction pipeline that supports folder-based batch testing
- Implement simple automatic hand detection using OpenCV
- Real-time recognition

2.0 Libraries Required

2.1 PyTorch <br>
model training and inference

2.2 torchvision <br>
transformations, datasets

2.3 numpy <br>
numerical operations

2.4 Pillow (PIL) <br>
image IO

2.5 OpenCV <br>
simple hand detection & cropping

2.6 python-dotenv <br>
Kaggle API credential loading

2.7 kaggle <br>
dataset downloading

2.8 scikit-learn<br>
evaluation metrics
</b>

### 3.0 Dataset(s)
Dataset 1 (Current): https://www.kaggle.com/datasets/prathumarikeri/american-sign-language-09az <br>
This one is good and offers from A-Z and 0-9 in Sign Language. Perfect for ML training with picture <br>
Posssible Dataset 2:https://universe.roboflow.com/sign-recognintion/sign-recoginition/dataset/1 <br>
This one is also very good and offers an even wider range of Sign Language, but it might be difficult to work with, and work better with pure AI instead. <br>
<br>

## 4.0 Integration 

### 4.1 Phase One — Dataset + Baseline Model
- Implemented automatic Kaggle dataset download / extraction
- Loaded dataset using PyTorch's ImageFolder
- Filtered classes to only 0–9 for proof of concept
- Built a simple Convolutional Neural Network
- Trained with adjustable epoch controls and live progress output

### 4.2 Phase Two — Prediction Pipeline
- Added predict.py to evaluate any image or folder of images
- Implemented sorted testing using /test/ directory
- Added basic hand cropping
  - Use of HSV for skin detection
- Try to solve issue when using real photos
- Reports per-image predictions and test accuracy
- Enables quick evaluation with real-world photos

### 4.3 Phase three
- Fixing bugs
- Add real-time implementation

### 4.4 Phase four
- Double-checking
- Fixing bufs
- Create a presentation

### 5.0 Firmware code   

### 5.1 main.py
stuff <br>

## 6.0 Acronyms
ASL — American Sign Language <br>
ML — Machine Learning <br>
CNN — Convolutional Neural Network <br>
HSV — Hue Saturation Value (color space used in skin detection) <br>

## References
Kaggle ASL Dataset:
https://www.kaggle.com/datasets/prathumarikeri/american-sign-language-09az <br>
Roboflow ASL Dataset:
https://universe.roboflow.com/sign-recognintion/sign-recoginition/dataset/1 <br>
PyTorch Documentation:
https://pytorch.org <br>
OpenCV Documentation:
https://docs.opencv.org <br>
