# 🌦 Weather Prediction using LSTM and XGBoost

This project implements a hybrid deep learning and machine learning model to forecast weather data. It combines **LSTM (Long Short-Term Memory)** networks for temporal feature extraction with **XGBoost** for final prediction, enabling accurate region-wise weather forecasting.

---

## 📌 Key Features

- Region-specific model training
- Sequence modeling using LSTM
- Feature-based prediction using XGBoost
- Unified pipeline for preprocessing, sequence generation, LSTM encoding, and final prediction
- Scikit-learn compatible pipeline for seamless integration and inference

---

## 🔁 Model Pipeline Overview

Each region undergoes the following steps:

1. **Data Preprocessing**
   - Normalization using `StandardScaler` (Scikit-learn)
   - Handling missing values, timestamp formatting, etc.

2. **Sequence Creation**
   - Transform flat time-series into sequences suitable for LSTM input

3. **LSTM Model**
   - Trains on time sequences to learn temporal features
   - Encoded features are extracted from the final LSTM hidden state

4. **XGBoost Training**
   - Takes the LSTM features as input and predicts the weather parameters (e.g., temperature, humidity)

5. **Unified Prediction Pipeline**
   - A reusable, exportable pipeline for each region
   - Includes preprocessing, LSTM encoding, and XGBoost prediction

> 🔁 This full pipeline is repeated independently for **each of the 11 regions**.




