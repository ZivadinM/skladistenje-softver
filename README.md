# Lung Cancer Dataset Analysis and Classification

## Overview
This project performs exploratory data analysis (EDA), data preprocessing, and classification modeling on a lung cancer dataset. The goal is to predict patient survival using machine learning models (XGBoost and Random Forest).

---

## Features

- **Dataset Loading**: Downloads and loads the lung cancer dataset.
- **Exploratory Data Analysis**: Uses Pandas Profiling to generate a detailed report.
- **Data Cleaning and Preprocessing**:
  - Handles missing data.
  - Converts date columns to calculate treatment duration.
  - Encodes categorical variables using one-hot encoding.
  - Converts cancer stages into ordinal numeric values.
- **Modeling**:
  - Splits data into train/test sets.
  - Trains and evaluates an XGBoost classifier.
  - Trains and evaluates a Random Forest classifier.
  - Shows confusion matrices and accuracy scores for both models.
  - Displays top 10 important features based on the Random Forest model.

---

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- kagglehub
- ydata-profiling
- scikit-learn
- xgboost

---

## Usage

1. **Download the dataset using Kagglehub:**

   ```python
   path = kagglehub.dataset_download("khwaishsaxena/lung-cancer-dataset")
   print("Path to dataset files:", path)
   ```

2. **Load the dataset:**

   ```python
    df = pd.read_csv('path_to/Lung Cancer.csv')
   ```

3. **Run the exploratory data analysis and profiling.**

4. **Preprocess the data and train models:**
    - Calculate treatment duration.
    - Encode categorical variables.
    - Convert cancer stages to ordinal numeric values.
    - Split data into train/test sets.
    - Train XGBoost and Random Forest models.
    - Evaluate model performance.
    - Display confusion matrices and accuracy scores.