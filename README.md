# California Housing Prices Prediction (R)

## Overview
This project analyzes the **California Housing dataset** to predict median house prices using regression-based models in **R**. The main goal is to compare multiple linear modeling approaches and identify the best-performing model based on predictive accuracy and model robustness.

The study focuses on:
- Understanding relationships between socioeconomic and geographic features
- Evaluating regularized regression techniques to handle multicollinearity
- Comparing model performance using appropriate evaluation metrics

---

## Models Implemented
The following linear models are implemented and compared:

- **Multiple Linear Regression (MLR)**  
  Baseline model using all predictors without regularization.

- **Ridge Regression**  
  Applies L2 regularization to reduce variance and mitigate multicollinearity.

- **Lasso Regression**  
  Applies L1 regularization, enabling feature selection by shrinking some coefficients to zero.

---

## Evaluation Metrics
Models are compared using:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Model interpretability and coefficient stability
