import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import save_model, load_model
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import joblib
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count, Manager
import psutil
import time
import gc
import signal

# Configure system for maximum performance with safety limits
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info messages
os.environ['OMP_NUM_THREADS'] = '1'  # Prevents conflicts with multiprocessing
tf.config.optimizer.set_jit(True)  # Enable XLA compilation for faster execution

# Set threads based on available cores, leaving some headroom
available_cores = max(1, cpu_count() - 2)
tf.config.threading.set_intra_op_parallelism_threads(available_cores)
tf.config.threading.set_inter_op_parallelism_threads(available_cores)

# Constants
SEQUENCE_LENGTH = 24
BATCH_SIZE = 256  # Larger batches for efficiency
MAX_MEMORY_USAGE = 0.8  # Max allowed memory usage before taking action
XGB_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.03,
    'max_depth': 7,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'random_state': 42,
    'n_jobs': max(1, available_cores),  # Use available cores for XGBoost
    'tree_method': 'hist',  # Faster training method
    'early_stopping_rounds': 15
}

def create_sequences(data, target, seq_length, step=1):
    """Create sequences with configurable step size"""
    X, y = [], []
    for i in range(0, len(data) - seq_length, step):
        X.append(data[i:i+seq_length])
        y.append(target[i+seq_length])
    return np.array(X), np.array(y)

def train_xgboost(X_train, X_train_lstm, y_train, X_test, X_test_lstm, y_test, target_vars):
    """Train XGBoost models in parallel for each target"""
    xgb_models = {}
    y_preds = np.zeros_like(y_test)
    
    X_train_combined = np.hstack((X_train.reshape(X_train.shape[0], -1), X_train_lstm))
    X_test_combined = np.hstack((X_test.reshape(X_test.shape[0], -1), X_test_lstm))
    
    for i, target in enumerate(target_vars):
        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X_train_combined, y_train[:, i], 
                 eval_set=[(X_test_combined, y_test[:, i])],
                 verbose=False)
        y_preds[:, i] = model.predict(X_test_combined)
        xgb_models[target] = model
    
    return y_preds, xgb_models

class RegionPipeline:
    def __init__(self, lstm_model, xgb_models, feature_scaler, target_scalers, features, target_variables):
        self.lstm_model = lstm_model
        self.xgb_models = xgb_models
        self.feature_scaler = feature_scaler
        self.target_scalers = target_scalers
        self.features = features  # Store original feature names
        self.target_variables = target_variables
        self.required_columns = ['timestamp', 'latitude', 'longitude', 'u10', 'v10', 'd2m', 
                               'msl', 'sp', 'tcc', 't2m', 'month_sin', 'month_cos']
        self.lag_columns = ['u10', 'v10', 'tcc', 't2m']
        
    def _add_temporal_features(self, df):
        """Add time-based features to new data"""
        if 'time' not in df.columns:
            raise ValueError("Input data must contain 'time' column")
            
        df['time'] = pd.to_datetime(df['time'])
        df['timestamp'] = df['time'].map(pd.Timestamp.timestamp)
        df['month_sin'] = np.sin(2 * np.pi * df['time'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['time'].dt.month / 12)
        return df
    
    def _calculate_lag_features(self, df):
        """Calculate lag and rolling features dynamically"""
        # Ensure we have enough history
        if len(df) < 3:
            raise ValueError(f"Need at least 3 timesteps for rolling features, got {len(df)}")
            
        # Calculate lag features
        for col in self.lag_columns:
            df[f'{col}_lag1'] = df[col].shift(1).fillna(df[col].mean())
            df[f'{col}_rolling_mean'] = df[col].rolling(window=3, min_periods=1).mean()
            df[f'{col}_rolling_std'] = df[col].rolling(window=3, min_periods=1).std().fillna(0)
        return df
    
    def preprocess_new_data(self, new_df):
        """Full preprocessing for new prediction data"""
        # 1. Add temporal features
        new_df = self._add_temporal_features(new_df)
        
        # 2. Calculate dynamic features
        new_df = self._calculate_lag_features(new_df)
        
        # 3. Ensure all required features are present
        missing = set(self.features) - set(new_df.columns)
        if missing:
            raise ValueError(f"Missing required features: {missing}")
            
        # 4. Scale features
        return self.feature_scaler.transform(new_df[self.features])
    
    def predict(self, new_data, sequence_length=24):
        """
        Args:
            new_data: DataFrame with raw input data (must include time column)
            sequence_length: Length of LSTM sequences
        Returns:
            Dictionary of {target: predictions}
        """
        # Preprocess the new data
        new_data_scaled = self.preprocess_new_data(new_data)
        
        # Create sequences
        X_new, _ = create_sequences(new_data_scaled, 
                                  np.zeros((len(new_data_scaled), len(self.target_variables))),
                                  sequence_length)
        
        # Get LSTM features
        lstm_features = self.lstm_model.predict(X_new)
        
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

def memory_safe_execution(func):
    """Decorator to monitor memory usage and clean up if needed"""
    def wrapper(*args, **kwargs):
        try:
            # Check memory before execution
            mem = psutil.virtual_memory()
            if mem.percent > 85:
                print(f"Warning: High memory usage before execution ({mem.percent}%)")
                gc.collect()
            
            result = func(*args, **kwargs)
            
            # Clean up after execution
            gc.collect()
            tf.keras.backend.clear_session()
            return result
        except MemoryError:
            print("Memory error detected, attempting cleanup...")
            gc.collect()
            tf.keras.backend.clear_session()
            return None
    return wrapper

def print_system_stats():
    """Print current system resource utilization"""
    mem = psutil.virtual_memory()
    print(f"\nCurrent System Stats:")
    print(f"CPU Usage: {psutil.cpu_percent()}%")
    print(f"Memory Usage: {mem.percent}%")
    print(f"Available Memory: {mem.available/1024/1024:.2f} MB")

def check_memory_usage():
    """Check if memory usage is within safe limits"""
    mem = psutil.virtual_memory()
    return mem.percent < (MAX_MEMORY_USAGE * 100)

def load_and_preprocess_data(filepath, target_variables):
    """Load and preprocess data in memory-efficient chunks"""
    print("Loading and preprocessing data...")
    start_time = time.time()
    
    # Load data in chunks if memory is tight
    if not check_memory_usage():
        print("Memory constrained, loading data in chunks...")
        chunks = []
        for chunk in pd.read_csv(filepath, parse_dates=['time'], infer_datetime_format=True, chunksize=100000):
            chunks.append(chunk)
        df = pd.concat(chunks, axis=0)
    else:
        df = pd.read_csv(filepath, parse_dates=['time'], infer_datetime_format=True)
    
    df = df.dropna(subset=['time'])
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna(subset=['time'])  # Drop any rows where time couldn't be parsed
    
    # Time features - use total seconds since epoch instead of timestamp()
    df['timestamp'] = (df['time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    
    # Cyclical month features
    df['month_sin'] = np.sin(2 * np.pi * df['time'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['time'].dt.month / 12)
    
    print(f"Data loaded in {time.time()-start_time:.2f} seconds")
    print_system_stats()
    return df

@memory_safe_execution
def create_features(df, target_vars):
    """Feature engineering with memory monitoring"""
    print("Creating features...")
    start_time = time.time()
    
    features = ['timestamp', 'latitude', 'longitude', 'u10', 'v10', 'd2m', 
               'msl', 'sp', 'tcc', 't2m', 'month_sin', 'month_cos']
    
    # Vectorized lag and rolling features in batches
    for col in ['u10', 'v10', 'tcc', 't2m']:
        df[f'{col}_lag1'] = df[col].shift(1).fillna(df[col].mean())
        
        # Process rolling features in chunks if memory is tight
        if not check_memory_usage():
            chunk_size = len(df) // 10
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i+chunk_size]
                df.loc[chunk.index, f'{col}_rolling_mean'] = chunk[col].rolling(3, min_periods=1).mean()
                df.loc[chunk.index, f'{col}_rolling_std'] = chunk[col].rolling(3, min_periods=1).std().fillna(0)
        else:
            df[f'{col}_rolling_mean'] = df[col].rolling(3, min_periods=1).mean()
            df[f'{col}_rolling_std'] = df[col].rolling(3, min_periods=1).std().fillna(0)
        
        features.extend([f'{col}_lag1', f'{col}_rolling_mean', f'{col}_rolling_std'])
    
    print(f"Features created in {time.time()-start_time:.2f} seconds")
    print_system_stats()
    return df, features

@memory_safe_execution
def process_region_batch(region_df, features, target_vars):
    """Process a batch of data for one region with memory safety"""
    print("Processing region batch...")
    start_time = time.time()
    
    # Normalize features and targets in chunks if needed
    feature_scaler = StandardScaler()
    
    if not check_memory_usage():
        chunk_size = len(region_df) // 10
        for i in range(0, len(region_df), chunk_size):
            region_df.iloc[i:i+chunk_size, region_df.columns.get_indexer(features)] = \
                feature_scaler.fit_transform(region_df.iloc[i:i+chunk_size][features])
    else:
        region_df[features] = feature_scaler.fit_transform(region_df[features])
    
    target_scalers = {}
    for target in target_vars:
        target_scaler = StandardScaler()
        region_df[target] = target_scaler.fit_transform(region_df[[target]])
        target_scalers[target] = target_scaler
    
    # Create sequences in smaller batches if memory is tight
    X, y = [], []
    data_values = region_df[features].values
    target_values = region_df[target_vars].values
    
    batch_size = BATCH_SIZE if check_memory_usage() else BATCH_SIZE // 2
    for i in range(0, len(data_values) - SEQUENCE_LENGTH, batch_size):
        batch_end = min(i + batch_size + SEQUENCE_LENGTH, len(data_values))
        X_batch, y_batch = create_sequences(data_values[i:batch_end], 
                                         target_values[i:batch_end], 
                                         SEQUENCE_LENGTH,step=2)
        X.append(X_batch)
        y.append(y_batch)
        # Clean up if memory is getting high
        if not check_memory_usage():
            gc.collect()
    
    X = np.concatenate(X) if len(X) > 0 else np.array([])
    y = np.concatenate(y) if len(y) > 0 else np.array([])
    
    print(f"Batch processed in {time.time()-start_time:.2f} seconds")
    print_system_stats()
    return X, y, feature_scaler, target_scalers

@memory_safe_execution
def train_lstm(X_train, y_train, sequence_length, num_features, num_targets):
    """Train LSTM with proper epoch counting"""
    # 1. Clear any previous session state
    tf.keras.backend.clear_session()
    gc.collect()

    # 2. Manual validation split (unchanged)
    val_size = int(0.2 * len(X_train))
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]

    # 3. Dataset pipeline (unchanged)
    batch_size = BATCH_SIZE if check_memory_usage() else BATCH_SIZE // 2
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
        .shuffle(1000) \
        .batch(batch_size) \
        .prefetch(tf.data.AUTOTUNE)
        
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)) \
        .batch(batch_size)

    # 4. Model construction with fresh initialization
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(sequence_length, num_features),
             dropout=0.3, recurrent_dropout=0.2),
        BatchNormalization(),
        LSTM(64, return_sequences=True, dropout=0.2),
        LSTM(32, dropout=0.2),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dense(num_targets)
    ])

    # 5. Force fresh weight initialization
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run()

    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])

    # 6. Verified epoch counter callback
    class EpochVerifier(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            print("\nTraining started - verifying epoch counting")
            
        def on_epoch_begin(self, epoch, logs=None):
            print(f"\nEpoch {epoch + 1} started (0-based index: {epoch})")

    print("Training LSTM model...")
    history = model.fit(
        train_dataset,
        epochs=100,
        validation_data=val_dataset,
        callbacks=[
            EpochVerifier(),
            ReduceLROnPlateau(patience=5, factor=0.5, verbose=1),
            EarlyStopping(patience=10, restore_best_weights=True)
        ],
        verbose=1,
        initial_epoch=0  # Explicitly start from epoch 0
    )
    
    return model, history

def process_region(region, region_df, target_vars):
    """Full processing pipeline for one region"""
    print(f"\nStarting processing for region {region} with {len(region_df)} samples")
    region_start = time.time()
    
    # Feature engineering
    region_df, features = create_features(region_df.copy(), target_vars)
    
    # Process in batches
    X, y, feature_scaler, target_scalers = process_region_batch(region_df, features, target_vars)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True)
    
    # Train LSTM
    lstm_model = train_lstm(X_train, y_train, SEQUENCE_LENGTH, len(features), len(target_vars))
    X_train_lstm = lstm_model.predict(X_train, batch_size=BATCH_SIZE)
    X_test_lstm = lstm_model.predict(X_test, batch_size=BATCH_SIZE)
    
    # Train XGBoost
    y_preds, xgb_models = train_xgboost(X_train, X_train_lstm, y_train, X_test, X_test_lstm, y_test, target_vars)
    

    # Create pipeline
    pipeline = RegionPipeline(
        lstm_model=lstm_model,
        xgb_models=xgb_models,
        feature_scaler=feature_scaler,
        target_scalers=target_scalers,
        features=features,
        target_variables=target_vars
    )
    
    # Evaluate
    results = []
    for i, target in enumerate(target_vars):
        y_pred_original = target_scalers[target].inverse_transform(y_preds[:, i].reshape(-1, 1))
        y_test_original = target_scalers[target].inverse_transform(y_test[:, i].reshape(-1, 1))
        
        results.append({
            'Region': region,
            'Target': target,
            'MAE': mean_absolute_error(y_test_original, y_pred_original),
            'MSE': mean_squared_error(y_test_original, y_pred_original),
            'RMSE': np.sqrt(mean_squared_error(y_test_original, y_pred_original)),
            'R2': r2_score(y_test_original, y_pred_original)
        })
    
    print(f"Completed region {region} in {time.time()-region_start:.2f} seconds")
    print_system_stats()
    
    return {
        'region': region,
        'pipeline': pipeline,
        'results': pd.DataFrame(results),
        'preprocessed_data': {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'features': features
        }
    }


def process_region_wrapper(region, region_df, target_vars, result_dict):
    """Wrapper function for region processing with error handling"""
    try:
        result = process_region(region, region_df.copy(), target_vars)
        result_dict[region] = result
    except Exception as e:
        print(f"Error processing region {region}: {str(e)}")
        result_dict[region] = {'error': str(e)}
    finally:
        # Force cleanup
        gc.collect()
        tf.keras.backend.clear_session()

def main():
    # Configuration
    filepath = "/mnt/c/Users/alanm/Downloads/Ernakulam_Data_1990_2024.csv"
    target_vars = ['u10', 'v10', 'd2m', 't2m', 'msl', 'sp', 'tcc']
    os.makedirs("region_results", exist_ok=True)
    
    print("Starting processing...")
    print(f"System has {cpu_count()} CPU cores, using {available_cores}")
    print_system_stats()
    
    # Load data
    df = load_and_preprocess_data(filepath, target_vars)
    df['num'] = df.get('num', np.random.randint(1, 12, size=len(df)))
    
    # Use Manager for shared results
    with Manager() as manager:
        result_dict = manager.dict()
        
        # Process regions with limited parallel processes
        max_workers = max(1, min(available_cores, 4))  # Don't use all cores
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for region in df['num'].unique():
                region_data = df[df['num'] == region].copy()
                futures.append(
                    executor.submit(
                        process_region_wrapper,
                        region,
                        region_data,
                        target_vars,
                        result_dict
                    )
                )
            
            # Monitor progress
            completed = 0
            total = len(futures)
            for future in as_completed(futures):
                completed += 1
                print(f"Completed {completed}/{total} regions")
                print_system_stats()
        
        # Convert managed dict to regular dict
        results = list(result_dict.values())
    
    # Save successful results
    successful_results = [r for r in results if not isinstance(r, dict) or 'error' not in r]
    if successful_results:
        all_pipelines = {r['region']: r['pipeline'] for r in successful_results if 'pipeline' in r}
        all_results = pd.concat([r['results'] for r in successful_results if 'results' in r])
        
        joblib.dump(all_pipelines, "all_region_pipelines.pkl")
        all_results.to_csv("combined_results_all_regions.csv", index=False)
    
    # Save error information
    errors = [r for r in results if isinstance(r, dict) and 'error' in r]
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(error['error'])
    
    print("\nProcessing complete")
    print_system_stats()

if __name__ == "__main__":
    # Handle keyboard interrupt gracefully
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    finally:
        # Clean up
        gc.collect()
        tf.keras.backend.clear_session()