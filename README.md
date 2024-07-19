# Lumbar_Spine_Degradation_Classification_FastAI
a classification pipeline developed to classify severe conditions of spine degradation from a dataset of 1983 people's MRI images. 

This project involves using machine learning techniques to classify lumbar spine degenerative diseases based on MRI scans. The focus is on detecting foraminal narrowing, subarticular stenosis, and canal stenosis in MRI scans of the lumbar spine and categorizing each condition's severity. The project utilizes FastAI for image processing and model training.

Project Overview
The project follows these main steps:

Data Preprocessing:

Convert 16-bit DICOM images to 8-bit PNG format for visualization and processing.
Upsample minority classes to ensure balanced training data.
Save preprocessed images and labels to a temporary directory.
Model Training:

Train convolutional neural network (CNN) models using FastAI's cnn_learner.
Utilize cross-validation to assess model performance.
Fine-tune the models and evaluate their accuracy.
Model Prediction:

Apply trained models to test data.
Save prediction results to CSV files for each column.
Visualization:

Display sample images from each category to validate the preprocessing steps.
Getting Started
Prerequisites
Ensure you have the following libraries installed:

Python 3.6+
FastAI
PyTorch
Pandas
NumPy
Matplotlib
Pydicom
OpenCV
Scikit-learn
tqdm
You can install the required libraries using pip:

bash
Copy code
pip install fastai pandas numpy matplotlib pydicom opencv-python scikit-learn tqdm
Data Preparation
Data Source: Download the dataset from Kaggle's RSNA 2024 Lumbar Spine Degenerative Classification competition.

Directory Structure: Place the dataset files in the following structure:



├── train.csv
├── train_series_descriptions.csv
├── train_images
│   ├── <study_id>
│   │   ├── <series_id>
│   │   │   ├── *.dcm
├── test_images
│   ├── <study_id>
│   │   ├── <series_id>
│   │   │   ├── *.dcm

Import Necessary Libraries:

python
Copy code
import os
import pandas as pd
import matplotlib.pyplot as plt
from fastai.vision.all import *
from sklearn.utils import resample
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import glob
import numpy as np
import pydicom
import cv2
from sklearn.model_selection import KFold

Preprocess Data:

python

def preprocess_data_for_column(train, column):
    # Implementation of the function
    # Refer to the provided code for details
    pass
    
Prepare Image Paths and Labels:

python

def prepare_image_paths_and_labels(upsampled_train, save_path, batch_size=1000, start_batch=0):
    # Implementation of the function
    # Refer to the provided code for details
    pass
    
Train Models:

python

def train_model_for_column(column, data, max_epochs=4):
    # Implementation of the function
    # Refer to the provided code for details
    pass

def cross_validate_model(data, column, max_epochs=4):
    # Implementation of the function
    # Refer to the provided code for details
    pass

def preprocess_and_train(df, columns, temp_image_dir, batch_size=1000, start_batch=0):
    # Implementation of the function
    # Refer to the provided code for details
    pass
    
Load Data and Merge with Series Descriptions:

python

train = pd.read_csv('/path/to/train.csv')
series_description = pd.read_csv('/path/to/train_series_descriptions.csv')
image_dir = "/path/to/train_images"

train = train.merge(series_description, on='study_id')
train = train[train['series_description'] == 'Sagittal T1']

Define Columns to Predict:

python

columns_to_predict = [
   list all the columns to predict
]

Process and Train Models:

python

temp_image_dir = "/path/to/temp_image_dir"
models = preprocess_and_train(train, columns_to_predict, temp_image_dir)

Predict on Test Data:

python

test_image_dir = "/path/to/test_images"

# Implementation of prediction on test data

Save Results:

python

for column, column_results in results.items():
    results_df = pd.DataFrame(column_results)
    output_path = f"/path/to/save/{column}_predictions.csv"
    results_df.to_csv(output_path, index=False)
    
Display Sample Images:

python

data2 = pd.concat([pd.read_csv(os.path.join(temp_image_dir, f"{column}_image_paths_labels.csv")) for column in columns_to_predict])
for category in ['Normal/Mild', 'Moderate', 'Severe']:
    show_images_from_category(data2, category, n=6)
    
Results and Evaluation
The results for each column are saved as CSV files, containing the study ID, series ID, image path, and predicted label. These results can be further analyzed and used for submission to the Kaggle competition.

Conclusion
This project demonstrates the application of machine learning techniques to medical imaging data, particularly for classifying lumbar spine degenerative diseases. By following the steps outlined in this README, you can reproduce the preprocessing, training, and prediction processes, and potentially improve the model's performance with further tuning and experimentation.

Acknowledgments
We acknowledge the data provided by RSNA and the resources available on Kaggle for this competition.
https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/data
