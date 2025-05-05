# Cancer-Prediction
# Cancer Prediction System
This is a cancer prediction system developed using machine learning techniques to predict the cancer stage and recommend a treatment plan for new patients based on their medical data.

## Table of Contents
Overview

Features

Requirements

Installation

Usage

Evaluation

## Overview
This system uses a Random Forest Classifier to predict the cancer stage based on various features like tumor size, lymph node involvement, menopause status, and more. Based on the predicted cancer stage, a treatment plan is recommended using manually defined rules.

The system's workflow includes:

Loading and preprocessing the dataset.

Training a model to predict the cancer stage.

Evaluating the model's performance with metrics such as accuracy, confusion matrix, and classification report.

Predicting cancer stage and recommending treatment for new patient data.

## Features
Cancer Stage Prediction: Predicts the cancer stage (Stage 1, Stage 2, Stage 3, Stage 4).

Treatment Recommendation: Based on the predicted cancer stage, the system recommends a corresponding treatment plan.

Data Preprocessing: Automatically handles missing values and encodes categorical features.

Evaluation: Evaluates the model's performance and visualizes the results using a confusion matrix.

## Requirements
Python 3.x

Libraries:

pandas

numpy

scikit-learn

matplotlib

You can install these dependencies using the following command:

pip install -r requirements.txt
Where the requirements.txt file contains:

pandas
numpy
scikit-learn
matplotlib
## Installation
Clone this repository to your local machine:

git clone https://github.com/your-username/cancer-prediction-system.git
Navigate to the project folder:

cd cancer-prediction-system
Install the required dependencies:

pip install -r requirements.txt
Ensure you have a CSV file (cancer.csv) with the relevant columns, such as age, menopause, tumor-size, inv-nodes, node-caps, deg-malig, breast, breast-quad, irradiat, and Class (cancer stage). Make sure the dataset is formatted correctly for the model to process.

## Usage
Run the script:
You can run the script from the command line as follows:

python cancer_prediction_system.py
Input: The system will require the medical data of a new patient as input. You can change the sample data in the script under new_patient_data for testing.

Example input:

new_patient_data = [60, 1, 4, 5, 1, 3, 0, 2, 0]  # Example values (age, menopause, tumor-size, inv-nodes, etc.)
You can modify these values with new patient information.

## Output:
The system will output:

Predicted cancer stage (Stage 1, Stage 2, Stage 3, or Stage 4).

Recommended treatment for the stage.

Example output:
t
--- Patient Prediction ---
🔍 Cancer Stage: Stage 2
💊 Recommended Treatment: Surgery + Radiation
## Evaluation
The system evaluates the trained model using the following:

Classification Report: Provides precision, recall, f1-score, and accuracy for cancer stage prediction.

Confusion Matrix: Visualizes the classification performance.

After training the model, the results will be shown for evaluation, including a confusion matrix plot.

