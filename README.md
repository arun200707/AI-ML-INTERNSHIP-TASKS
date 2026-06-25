House Price Prediction using Machine Learning
Project Overview

This project builds a complete Machine Learning pipeline to predict house prices based on various property features. The objective is to compare multiple regression algorithms, identify the best-performing model, analyze important features affecting house prices, and provide actionable insights through model evaluation.

Problem Statement

Real estate prices are influenced by several factors such as area, number of bedrooms, location, and amenities. Accurately predicting house prices helps:

Home buyers make informed decisions.
Real estate companies estimate property values.
Investors identify profitable opportunities.
Banks assess property valuations for loans.

The goal of this project is to develop a predictive model that estimates house prices using historical housing data.

Dataset Description

The dataset contains housing-related features such as:

Feature	Description
Area	Total area of the house (sq ft)
Bedrooms	Number of bedrooms
Bathrooms	Number of bathrooms
Stories	Number of floors
Parking	Number of parking spaces
Main Road	Access to main road
Guest Room	Availability of guest room
Basement	Availability of basement
Hot Water Heating	Hot water system availability
Air Conditioning	Air conditioning availability
Furnishing Status	Furnished/Semi-Furnished/Unfurnished
Price	Target variable
Project Pipeline
1. Data Collection
Load housing dataset using Pandas.
Inspect dataset dimensions and feature types.
df = pd.read_csv("Housing.csv")
2. Data Preprocessing
Handling Missing Values
df.isnull().sum()
Encoding Categorical Variables
df = pd.get_dummies(df, drop_first=True)
Feature Selection

Separate independent and dependent variables.

X = df.drop("price", axis=1)
y = df["price"]
Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
3. Model Training

Three different regression algorithms were trained and evaluated.

Model 1: Linear Regression
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
Model 2: Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
Model 3: Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)
Model Evaluation

The models were evaluated using:

R² Score
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
from sklearn.metrics import r2_score
Results Comparison
Model	R² Score
Linear Regression	0.65
Decision Tree	0.72
Random Forest	0.85

(Values are sample results and may vary depending on the dataset.)

Best Model Selection
Why Random Forest?

Random Forest was selected as the best model because:

Highest R² score.
Handles non-linear relationships effectively.
Reduces overfitting through ensemble learning.
Captures complex feature interactions.
Provides feature importance scores.

Therefore, Random Forest delivers the most accurate house price predictions.

Feature Importance Analysis

Feature importance helps identify which factors most strongly influence house prices.

import matplotlib.pyplot as plt

importance = rf.feature_importances_

feat_importance = pd.Series(
    importance,
    index=X.columns
).sort_values(ascending=False)

feat_importance.plot(
    kind="bar",
    figsize=(10,6)
)

plt.title("Feature Importance")
plt.ylabel("Importance Score")
plt.show()
Key Influential Features

The most important features typically include:

Area
Bathrooms
Air Conditioning
Number of Stories
Parking Spaces

These features contribute significantly to property value.

Prediction Example
new_house = [[5000,4,3,2,2]]

prediction = rf.predict(new_house)

print("Predicted Price:", prediction[0])
Project Structure
House-Price-Predictor/
│
├── Housing.csv
├── house_price_prediction.ipynb
├── README.md
├── requirements.txt
└── feature_importance.png
Technologies Used
Python
Pandas
NumPy
Matplotlib
Scikit-Learn
Jupyter Notebook
Executive Summary
Business Objective

Develop a machine learning solution capable of accurately estimating house prices based on property characteristics.

Approach

Three regression algorithms were implemented:

Linear Regression
Decision Tree Regressor
Random Forest Regressor

The dataset was cleaned, encoded, split into training and testing sets, and evaluated using R² score and error metrics.

Key Findings
Random Forest achieved the highest predictive accuracy.
Property area was the most influential feature.
Ensemble learning significantly improved performance over individual models.
Conclusion

The Random Forest model is recommended for deployment due to its superior predictive capability and robustness. Feature importance analysis revealed that house size, amenities, and structural characteristics are the primary drivers of housing prices.

Future Improvements
Hyperparameter tuning using GridSearchCV.
Cross-validation for robust evaluation.
Deployment using Flask API.
Integration with a web application.
Use of larger real-estate datasets for better generalization.
