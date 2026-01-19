# Retail Store Performance Analysis and Sales Prediction

This repository contains a comprehensive data science project analyzing the key drivers of retail store performance. The study utilizes statistical modeling and machine learning techniques to predict **Monthly Sales Revenue** and evaluate the impact of various operational and demographic factors.

## Project Overview

This project aims to:

1. **Predict Revenue:** Develop robust models to forecast monthly sales based on store features.
2. **Analyze Drivers:** Identify which variables have the most significant statistical impact on performance.
3. **Segment Stores:** Explore natural groupings within the data to test segmentation hypotheses.
4. **Compare Models:** Evaluate the performance of linear versus non-linear approaches to understand the nature of the underlying data relationships.

## Dataset Information

The analysis is performed on a **synthetic dataset** designed to simulate real-world retail scenarios. However, because of that, some of preprocessing parts are not needed, and not done. Also results are surprising for real life scenarios.

* **Observations:** 1,650 Retail Stores.
* **Features:** The dataset includes a mix of numerical and categorical variables, including:
* **Operational:** Store Size, Product Variety, Employee Efficiency, Store Age.
* **Marketing:** Marketing Spend, Promotions Count.
* **External:** Competitor Distance, Economic Indicators, Customer Footfall.
* **Location:** City, Store Category.

## Methodology

The project follows a structured data science pipeline:

### 1. Data Preprocessing and EDA

* **Data Cleaning:** Handling missing values and outlier detection using boxplot analysis.
* **Multicollinearity Check:** Analysis of independent variables using Correlation Matrices and Variance Inflation Factor (VIF) to ensure model stability. High-collinearity features were removed or transformed.
* **Feature Engineering:**
* **Scaling:** Implementation of Min-Max normalization and Z-Score standardization depending on algorithm requirements.
* **Encoding:** Application of One-Hot Encoding and Dummy Coding () for categorical variables.



### 2. Unsupervised Learning (Clustering)

* **K-Means Clustering:** Applied to segment stores based on performance and operational metrics.
* **Dimensionality Reduction:** **Principal Component Analysis (PCA)** was utilized to visualize high-dimensional cluster separations in a 2D space.
* **Hypothesis Generation:** Derived cluster labels were integrated into supervised models to test if segmentation provides additional predictive power.

### 3. Supervised Learning (Predictive Modeling)

Two distinct algorithms were trained and compared to determine the best fit for the data:

* **Linear Regression (OLS):** Used to establish a baseline and interpret the direct linear effect (coefficients) of each variable on sales.
* **Random Forest:** Implemented to capture potential non-linear relationships and complex interactions between variables. Hyperparameters (such as `mtry`) were tuned for optimization.
