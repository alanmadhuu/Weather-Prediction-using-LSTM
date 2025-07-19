import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Union
from sklearn.preprocessing import StandardScaler
import re
import logging


# Define all required classes exactly as during training
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
        x = x[:, -1, :]  # Take last timestep only
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class RegionPipeline:
    def __init__(self, lstm_model, xgb_models, feature_scaler, target_scalers, features, target_variables):
        self.lstm_model = lstm_model
        self.xgb_models = xgb_models
        self.feature_scaler = feature_scaler
        self.target_scalers = target_scalers
        self.features = features
        self.target_variables = target_variables
        self.required_columns = ['timestamp', 'latitude', 'longitude', 'u10', 'v10', 'd2m', 
                               'msl', 'sp', 'tcc', 't2m', 'month_sin', 'month_cos']
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
        # Use proper windowing with forward filling
        for col in self.lag_columns:
            df[f'{col}_lag1'] = df[col].shift(1).fillna(method='ffill')
            df[f'{col}_rolling_mean'] = df[col].rolling(3, min_periods=1).mean().fillna(method='ffill')
            df[f'{col}_rolling_std'] = df[col].rolling(3, min_periods=1).std().fillna(0)
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
        
        # Ensure we have enough data points
        if len(new_data_scaled) < sequence_length:
            raise ValueError(f"Need at least {sequence_length} timesteps, got {len(new_data_scaled)}")
        
        # Create sequences with proper shape (batch_size, seq_len, num_features)
        X_new = np.array([new_data_scaled[i:i+sequence_length] for i in range(len(new_data_scaled)-sequence_length+1)])
        
        # Get LSTM features
        with torch.no_grad():
            self.lstm_model.eval()
            X_new_tensor = torch.FloatTensor(X_new).to(self.device)
            lstm_features = self.lstm_model(X_new_tensor).cpu().numpy()
        
        # Combine features
        combined_features = np.hstack((
            X_new.reshape(X_new.shape[0], -1),  # Flatten the sequence
            lstm_features
        ))
        
        # Make predictions
        predictions = {}
        for target in self.target_variables:
            pred_scaled = self.xgb_models[target].predict(combined_features)
            predictions[target] = self.target_scalers[target].inverse_transform(
                pred_scaled.reshape(-1, 1)).flatten()
        return predictions

def create_sequences(data, target, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)

class WeatherPredictionSystem:
    def __init__(self, model_path: str, region_num=1):
        self.model_path = model_path
        self.region_num = region_num
        self.pipeline = None
        self.time_step_hours = 1
        self.sequence_length = 24
        self.target_variables = ['u10', 'v10', 'd2m', 't2m', 'msl', 'sp', 'tcc']
        self._load_model()


    def _extract_region_num(self):
        match = re.search(r'region_(\d+)\.pkl$', self.model_path)
        if not match:
            raise ValueError(f"Invalid model path format: {self.model_path}")
        return int(match.group(1))    
    
    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
        try:
            # Load all regions pipelines once
            if not hasattr(self, 'all_pipelines'):
                WeatherPredictionSystem.all_pipelines = joblib.load(self.model_path)
                print("Successfully loaded all region pipelines")
            
            # Get the specific region pipeline
            self.pipeline = WeatherPredictionSystem.all_pipelines.get(self.region_num)
            
            if not self.pipeline:
                raise ValueError(f"No pipeline found for region {self.region_num}")
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    
    def get_available_locations(self) -> List[int]:
        return [1]

    def predict_weather(self, timestamp: Union[datetime, str]) -> Dict[str, float]:
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except ValueError:
                raise ValueError("Timestamp must be in ISO format (YYYY-MM-DDTHH:MM:SS)")

        historical_data = self._create_historical_data(timestamp)
        try:
            predictions = self.pipeline.predict(historical_data)
            return {var: float(predictions[var][-1]) for var in self.target_variables}
        except Exception as e:
            print(f"Prediction failed: {str(e)}")
            raise
    
    def _create_historical_data(self, timestamp: datetime) -> pd.DataFrame:
        OG_DATA_PATH = r"C:\Users\alanm\Downloads\Ernakulam_Data_1990_2025.csv"
        
        # Load and filter data
        og_df = pd.read_csv(
            OG_DATA_PATH,
            parse_dates=['time'],
            dayfirst=True,
            date_format='%d-%m-%Y %H:%M',
            low_memory=False
        )
        
        # Get coordinates for current region
        region_info = og_df[og_df['num'] == self.region_num][['latitude', 'longitude']].drop_duplicates()
        if not region_info.empty:
            region_lat = region_info['latitude'].iloc[0]
            region_lon = region_info['longitude'].iloc[0]
        else:
            # Fallback coordinates based on region number
            region_lat = 9.98 + (self.region_num * 0.01)
            region_lon = 76.28 + (self.region_num * 0.01)
        
        # Filter for current region
        region_df = og_df[(og_df['num'] == self.region_num) & (og_df['time'] <= timestamp)].copy()
        region_df = region_df.sort_values('time').reset_index(drop=True)

        # Generate realistic synthetic data if needed
        if len(region_df) < self.sequence_length:
            missing = self.sequence_length - len(region_df)
            synth_data = []
            last_valid = region_df.iloc[-1].copy() if not region_df.empty else None
            
            for i in range(missing):
                if last_valid is not None:
                    synth_row = last_valid.copy()
                else:
                    # Initialize with region 1 defaults
                    synth_row = pd.Series({
                        'num': 1,
                        'latitude': region_lat,
                        'longitude': region_lon,
                        'u10': 1.0 + np.random.uniform(-0.2, 0.2),
                        'v10': 0.5 + np.random.uniform(-0.1, 0.1),
                        't2m': 300.0 + np.random.normal(0, 2),
                        'd2m': 298.0 + np.random.normal(0, 2),
                        'msl': 101325.0 * np.random.uniform(0.999, 1.001),
                        'sp': 100000.0 * np.random.uniform(0.995, 1.005),
                        'tcc': np.clip(0.3 + np.random.uniform(-0.1, 0.1), 0, 1),
                    })

                # Set time with temporal variations
                time_offset = (missing - i) * self.time_step_hours
                synth_row['time'] = timestamp - timedelta(hours=time_offset)
                
                # Add realistic temporal patterns
                hour = synth_row['time'].hour
                synth_row['t2m'] = 300 + 5 * np.sin(2 * np.pi * hour/24) + np.random.normal(0, 0.5)
                synth_row['d2m'] = synth_row['t2m'] - np.random.uniform(1, 3)
                
                synth_row['u10'] = synth_row.get('u10', 1.0) * np.random.uniform(0.95, 1.05)
                synth_row['v10'] = synth_row.get('v10', 0.0) * np.random.uniform(0.95, 1.05)
                synth_row['msl'] = synth_row.get('msl', 101325) * np.random.uniform(0.9998, 1.0002)
                synth_row['tcc'] = np.clip(0.3 + 0.4 * np.sin(2 * np.pi * hour/24) + np.random.uniform(-0.1, 0.1), 0, 1)
                
                synth_data.append(synth_row)
                last_valid = synth_row.copy()
            
            synth_df = pd.DataFrame(synth_data)
            region_df = pd.concat([region_df, synth_df], ignore_index=True)

        # Ensure required columns exist (now handled during synthetic data generation)
        return region_df.tail(self.sequence_length)

# Define a global dictionary to store WeatherPredictionSystem instances
weather_systems = {}

# Define a dummy app object with a logger
class DummyApp:
    def __init__(self):
        self.logger = logging.getLogger("WeatherPredictionSystem")
        self.logger.setLevel(logging.DEBUG)

app = DummyApp()

def get_weather_system(region_num):
    if region_num not in weather_systems:
        try:
            weather_systems[region_num] = WeatherPredictionSystem(
                MODEL_PATH,  # Same path for all regions
                region_num=region_num  # Pass region number directly
            )
            app.logger.info(f"Initialized prediction system for region {region_num}")
        except Exception as e:
            app.logger.error(f"Error initializing region {region_num}: {str(e)}")
            return None
    return weather_systems[region_num]

# Example usage
if __name__ == "__main__":
    # Load the specific region 1 model
    MODEL_PATH = r"C:\Users\alanm\Desktop\mini project\sample\main\region_results\all_region_pipelines.pkl"
    weather_system = WeatherPredictionSystem(MODEL_PATH)
    
    # Get available locations (should only be [1])
    print("Available locations:", weather_system.get_available_locations())
    
    # Example prediction with timestamp input
    try:
        result = weather_system.predict_weather(
            timestamp="2025-04-03 12:00:00"  # ISO format string
        )
        print("\nPrediction results for region 1:")
        for var, value in result.items():
            print(f"{var}: {value:.2f}")
    except Exception as e:
        print(f"Error making prediction: {str(e)}")