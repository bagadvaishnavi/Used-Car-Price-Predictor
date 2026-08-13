# 🚗 Used Car Price Predictor

## 📌 Project Overview

Used Car Price Predictor is a machine learning project developed using Python to estimate the selling price of a used car based on various car-related features.

The project uses Random Forest Regression to learn the relationship between car characteristics and historical selling prices.

## 🎯 Objectives

- Predict the estimated selling price of a used car.
- Apply machine learning to a real-world regression problem.
- Perform data preprocessing and categorical encoding.
- Evaluate the performance of the machine learning model.
- Develop a simple GUI for user-friendly price prediction.

## 📊 Dataset

The dataset contains information about used cars including:

- Manufacturing Year
- Present Price
- Kilometers Driven
- Fuel Type
- Seller Type
- Transmission
- Previous Owners
- Selling Price

## 🤖 Machine Learning Algorithm

The project uses:

**Random Forest Regression**

Random Forest combines multiple decision trees to produce a robust prediction.

## 🔧 Technologies Used

### Programming Language
- Python

### Libraries
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Tkinter

## ⚙️ Project Workflow

1. Dataset Collection
2. Data Exploration
3. Data Preprocessing
4. Feature and Target Separation
5. Categorical Encoding
6. Train-Test Split
7. Random Forest Model Training
8. Model Prediction
9. Model Evaluation
10. Feature Importance Analysis
11. GUI Development

## 📈 Model Performance

The Random Forest Regression model achieved:

**R² Score: 0.9652**

**Mean Squared Error: approximately 0.8009**

The feature importance analysis showed that Present Price was the most influential feature in the prediction.

## 🖥️ Application

A graphical user interface was developed using Tkinter.

The user can enter:

- Manufacturing Year
- Present Price
- Kilometers Driven
- Fuel Type
- Seller Type
- Transmission
- Previous Owners

The application then displays the estimated selling price.

## 📁 Project Structure

```text
Used-Car-Price-Predictor/
│
├── data/
│   └── car data.csv
│
├── app.py
├── test_data.py
├── car_price_model.pkl
├── model_columns.pkl
├── feature_importance.png
└── README.md
