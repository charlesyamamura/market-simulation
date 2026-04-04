# Forecasting New Product Concepts and Markets: Domain-Integrated Machine Learning

XGBoost | Random Forest | PyTorch Deep Learning | PCA
This repository implements a specialized forecasting and design methodology that merges industry domain expertise with machine learning. By utilizing advanced feature engineering, this approach achieves high predictive accuracy on specialized datasets where traditional "Big Data" is unavailable.
XGBoost, Random Forest, and Deep Learning models are presented. 
A Principal Component Analysis model is used to interpret, understand, and explain market and product features.
Peer reviewed scientific articles explaining the method in detail are also posted.

## Project Overview

Domain-Centric Engineering: Leverages industry knowledge to craft features that preclude the need for massive datasets.
Multi-Model Pipeline: A comparative analysis of Gradient Boosting (XGBoost), Ensemble Learning (Random Forest), and Deep Learning (PyTorch MLP).
Hardware Acceleration: Optimized for Apple Silicon (M1/M2/M3) using the MPS (Metal Performance Shaders) backend.
Interpretability: Integrates Principal Component Analysis (PCA) and Feature Importance to decode complex market drivers.

## 🧠 Model Summary

XGBoost
- Strategy: Decision trees with Gradient Boosting, optimized via TimeSeriesSplit cross-validation.
- Best parameters: {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200, 'subsample': 0.8}

Random Forest
- Strategy: Ensemble of 200–400 estimators with square-root feature selection.
- Insight: Proved the most robust against temporal shifts (2018 $\rightarrow$ 2019 split).
- Best parameters: {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}

Deep Learning (PyTorch MLP)
- Architecture: 2-Layer Deep Neural Network with LeakyReLU activations.
- Optimization: Adam optimizer with Weight Decay and Dropout (0.2) to prevent overfitting.
- Safeguards: Implements a custom Early Stopping class to restore the best model state across epochs.

### Performance (model / database / R2 / RMSE)
Random Forest    Train 0.900394  0.246200
Random Forest    Test  0.773142  0.395121
XGBoost          Train 0.900964  0.245493
XGBoost          Test  0.701305  0.453384
Neural Network   Train 0.886319  0.263019
Neural Network   Test  0.710101  0.446659

## 🔧 Setup
This code is optimized for Apple Silicon. It automatically detects the mps device for accelerated training on Mac M1/M2/M3 chips.

### Dependencies

```bash
pip install torch pandas numpy scikit-learn matplotlib xgboost openpyxl
