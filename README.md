# Cancer Diagnosis with PySpark MLlib
# Project Overview
This project focuses on diagnosing cancer using machine learning algorithms in PySpark's MLlib. The dataset comprises clinical variables related to cancer diagnosis and is classified into two categories: Benign and Malignant. The project involves data preprocessing, training two machine learning models (Random Forest and Logistic Regression), and evaluating their performances.

# Environment Setup
Ensure Python and PySpark are installed in your environment. The project uses environment variables for PySpark configuration.

# Dataset
The dataset is initially in a CSV format with clinical variables and needs to be converted into the LibSVM format for use with Spark MLlib.

## Files
- project3_data.csv: Original dataset in CSV format.
- project3_data_clean.txt: Processed dataset in LibSVM format.

## Implementation Details
- Data Conversion: A Python function converts the dataset from CSV to LibSVM format.
- Model Training: Two models, Random Forest and Logistic Regression, are trained on the dataset.
- Model Evaluation: Models are evaluated based on Accuracy, F1 Score, Precision, Recall, and Area under ROC.

## Usage
Data Conversion: Replace the path of the script, run the script to convert the CSV data to LibSVM format.
Model Training and Evaluation: Load the LibSVM file to train the models and evaluate their performances.

## Dependencies
- PySpark
- Pandas

## Running the Project
Set up the PySpark environment, and have the project3_data.csv file and the script under the same directory,
Execute the Python script.
Review the output for model evaluations.

## Author
Long Huang
