import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Union
from sklearn.preprocessing import StandardScaler

def create_sequences(data, target, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)

# LSTM Model (unchanged)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64, hidden_size3=32, output_size=7):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size1,
            num_layers=3,
            batch_first=True,
            dropout=0.3
        )
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size1, 32)
        self.fc2 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Region Pipeline (modified prediction logic)
class RegionPipeline:
    def __init__(self, lstm_model, xgb_models, feature_scaler, target_scalers, features, target_variables):
        self.lstm_model = lstm_model
        self.xgb_models = xgb_models
        self.feature_scaler = feature_scaler
        self.target_scalers = target_scalers
        self.features = features
        self.target_variables = target_variables
        self.required_columns = ['time', 'latitude', 'longitude', 'u10', 'v10', 'd2m', 
                               'msl', 'sp', 'tcc', 't2m']
        self.lag_columns = ['u10', 'v10', 'tcc', 't2m']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _add_temporal_features(self, df):
        if 'time' not in df.columns:
            raise ValueError("Input data must contain 'time' column")
            
        df['time'] = pd.to_datetime(df['time'])
        df['timestamp'] = df['time'].map(pd.Timestamp.timestamp)
        df['month_sin'] = np.sin(2 * np.pi * df['time'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['time'].dt.month / 12)
        return df
    
    def _calculate_lag_features(self, df):
        if len(df) < 3:
            raise ValueError(f"Need at least 3 timesteps for rolling features, got {len(df)}")
            
        for col in self.lag_columns:
            df[f'{col}_lag1'] = df[col].shift(1).fillna(df[col].mean())
            # Fixed the typo here (changed 'col' to col)
            df[f'{col}_rolling_mean'] = df[col].rolling(window=3, min_periods=1).mean()
            df[f'{col}_rolling_std'] = df[col].rolling(window=3, min_periods=1).std().fillna(0)
        return df
    
    def preprocess_new_data(self, new_df):
        new_df = self._add_temporal_features(new_df)
        new_df = self._calculate_lag_features(new_df)
        
        missing = set(self.features) - set(new_df.columns)
        if missing:
            raise ValueError(f"Missing required features: {missing}")
            
        return self.feature_scaler.transform(new_df[self.features])
    
    def predict(self, new_data, sequence_length=24):
        new_data_scaled = self.preprocess_new_data(new_data)
        
        # Create sequences
        X_new, _ = create_sequences(new_data_scaled, 
                                  np.zeros((len(new_data_scaled), len(self.target_variables))),
                                  sequence_length)
        
        # Get LSTM features
        with torch.no_grad():
            self.lstm_model.eval()
            X_new_tensor = torch.FloatTensor(X_new).to(self.device)
            lstm_features = self.lstm_model(X_new_tensor).cpu().numpy()
        
        # Combine features
        combined_features = np.hstack((
            X_new.reshape(X_new.shape[0], -1),
            lstm_features
        ))
        
        # Make predictions
        predictions = {}
        for target in self.target_variables:
            pred_scaled = self.xgb_models[target].predict(combined_features)
            predictions[target] = self.target_scalers[target].inverse_transform(
                pred_scaled.reshape(-1, 1)).flatten()
        
        return predictions

# Weather Prediction System (modified)
class WeatherPredictionSystem:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.pipeline = None
        self.time_step = 6  # 6-hour intervals
        self.sequence_length = 24
        self.target_vars = ['u10', 'v10', 'd2m', 't2m', 'msl', 'sp', 'tcc']
        self._load_model()
    
    def _load_model(self):
        if not os.path.exists(self.model_path):     
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        self.pipeline = joblib.load(self.model_path)
        print(f"Loaded model for region 1")

    def get_available_locations(self) -> List[int]:
        return [1]

    def predict_weather(self, start_time: Union[datetime, str], hours_ahead: int = 6) -> List[Dict[str, float]]:
        # Validate input
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time)
        
        # Load historical data
        hist_data = self._get_historical_data(start_time)
        if len(hist_data) < self.sequence_length:
            raise ValueError(f"Need minimum {self.sequence_length} historical observations")
        
        # Prepare initial window
        current_data = hist_data.tail(self.sequence_length).copy()
        predictions = []
        
        # Calculate number of steps needed
        steps = hours_ahead // self.time_step
        for _ in range(steps):
            # Predict next step
            try:
                pred = self.pipeline.predict_next(current_data)
            except Exception as e:
                raise RuntimeError(f"Prediction failed at step {_+1}: {str(e)}")
            
            # Record prediction
            predictions.append(pred)
            
            # Create new row for autoregression
            new_time = current_data['time'].iloc[-1] + timedelta(hours=self.time_step)
            new_row = {
                'time': new_time,
                'latitude': current_data['latitude'].iloc[-1],
                'longitude': current_data['longitude'].iloc[-1],
                **{k: v for k, v in pred.items() if k in current_data.columns}
            }
            
            # Append prediction to data
            current_data = pd.concat([
                current_data, 
                pd.DataFrame([new_row])
            ], ignore_index=True).tail(self.sequence_length)
        
        return predictions

    def _get_historical_data(self, end_time: datetime) -> pd.DataFrame:
        DATA_PATH = r"C:\Users\alanm\Downloads\Ernakulam_Data_1990_2025.csv"
        df = pd.read_csv(
            DATA_PATH,
            parse_dates=['time'],
            date_format='%d-%m-%Y %H:%M',
            low_memory=False
        )
        return df[
            (df['num'] == 1) & 
            (df['time'] <= end_time)
        ].sort_values('time').reset_index(drop=True)

# Example Usage
if __name__ == "__main__":
    system = WeatherPredictionSystem(r"C:\Users\alanm\Desktop\mini project\sample\main\region_results\pipeline_region_1.pkl")
    
    try:
        # Predict 24 hours ahead (4 steps x 6 hours)
        results = system.predict_weather(
            start_time="2025-03-31 12:00:00",
            hours_ahead=24
        )
        
        print("\nWeather Predictions:")
        for i, pred in enumerate(results):
            print(f" +{(i+1)*6} hours:")
            for var, val in pred.items():
                print(f" {var}: {val:.2f}")
    except Exception as e:
        print(f"Prediction error: {str(e)}")