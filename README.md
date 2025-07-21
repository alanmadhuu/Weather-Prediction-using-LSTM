# 🌦 Weather Prediction using LSTM and XGBoost

This project implements a hybrid deep learning and machine learning system to forecast weather conditions. It combines **LSTM (Long Short-Term Memory)** networks for temporal sequence learning with **XGBoost** for accurate prediction.

A **Django-based Python backend** handles the model logic and exposes APIs, while a simple **HTML frontend** provides a user-friendly interface for interaction.

---

## 📌 Key Features

- Region-specific weather prediction (11 regions)
- LSTM for temporal sequence modeling
- XGBoost for final prediction
- End-to-end unified prediction pipeline per region
- Web application with Django backend and HTML frontend
- Modular code for easy extension or region scaling

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




