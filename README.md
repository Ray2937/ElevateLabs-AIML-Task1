# Task 1: Data Cleaning & Preprocessing

## Overview
This repository contains the solution for Task 1 of the AIML Internship at Elevate Labs. The goal is to clean, transform, and prepare raw data for machine learning models using the Titanic dataset.

## Files Included
- `task1.py`: Python script containing the entire preprocessing pipeline.
- `Titanic-Dataset.csv`: The original raw dataset.
- `Titanic_Cleaned.csv`: The final processed and cleaned dataset.
- `fare_boxplot_before.png`: Visualization of outliers in the 'Fare' column before cleaning.
- `fare_boxplot_after.png`: Visualization of the 'Fare' column after removing outliers.

## Steps Performed
1. **Exploratory Data Analysis (EDA):** Loaded the dataset, examined its shape, data types, and checked for missing values.
2. **Missing Value Handling:** Filled missing `Age` values with the median, filled missing `Embarked` values with the mode, and dropped the `Cabin` column due to a high percentage of missing data.
3. **Categorical Encoding:** Converted the `Sex` column into binary numerical values and applied One-Hot Encoding to the `Embarked` column.
4. **Feature Scaling:** Standardized numerical features (`Age` and `Fare`) using Scikit-Learn's `StandardScaler`.
5. **Outlier Detection & Removal:** Used boxplots and the Interquartile Range (IQR) method to detect and remove extreme outliers from the `Fare` column.