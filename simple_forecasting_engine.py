import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class SimpleForecastingEngine:
    """Simplified forecasting engine with core ML models only"""

    def __init__(self):
        self.models = {}
        self.predictions = {}
        self.metrics = {}

    def train_linear_regression(self, X_train, y_train, X_test, y_test):
        """Train Linear Regression model"""
        model = LinearRegression()
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        self.models['Linear Regression'] = model
        self.predictions['Linear Regression'] = {
            'train': train_pred,
            'test': test_pred
        }

        return model, train_pred, test_pred

    def train_random_forest(self, X_train, y_train, X_test, y_test, n_estimators=100):
        """Train Random Forest model"""
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        self.models['Random Forest'] = model
        self.predictions['Random Forest'] = {
            'train': train_pred,
            'test': test_pred
        }

        return model, train_pred, test_pred

    def train_neural_network(self, X_train, y_train, X_test, y_test, max_iter=500):
        """Train Neural Network model using scikit-learn MLPRegressor"""
        model = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=max_iter,
            random_state=42
        )
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        self.models['Neural Network'] = model
        self.predictions['Neural Network'] = {
            'train': train_pred,
            'test': test_pred
        }

        return model, train_pred, test_pred

    def calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2
        }

    def evaluate_models(self, y_train, y_test):
        """Evaluate all trained models"""
        results = {}

        for model_name, predictions in self.predictions.items():
            train_metrics = self.calculate_metrics(y_train, predictions['train'])
            test_metrics = self.calculate_metrics(y_test, predictions['test'])

            results[model_name] = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics
            }

        self.metrics = results
        return results

    def forecast_future(self, model_name, steps=30, last_sequence=None):
        """Make future predictions"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train the model first.")

        model = self.models[model_name]

        if model_name == 'Neural Network':
            if last_sequence is None:
                raise ValueError("Last sequence is required for Neural Network forecasting")
            future_pred = []
            current_sequence = last_sequence.copy()

            for _ in range(steps):
                pred = model.predict([current_sequence])
                future_pred.append(pred[0])
                # Update sequence for next prediction
                current_sequence = np.append(current_sequence[1:], pred[0])

            future_pred = np.array(future_pred)
        else:
            # For other ML models, create a simple trend-based forecast
            if last_sequence is None:
                raise ValueError("Last sequence is required for forecasting")

            # Simple approach: use the last known trend
            recent_trend = np.mean(np.diff(last_sequence[-5:]))
            last_value = last_sequence[-1]
            future_pred = np.array([last_value + (i + 1) * recent_trend for i in range(steps)])

        return future_pred

    def prepare_data(self, df, date_col, target_col):
        """Prepare data for time series analysis"""
        # Convert date column to datetime
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        # Set date as index and sort
        df = df.set_index(date_col).sort_index()

        # Remove missing values
        df = df.dropna()

        return df

    def create_sequences(self, data, seq_length):
        """Create sequences for ML training"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    def auto_forecast(self, df, date_col, target_col, forecast_steps=30):
        """Automatically find the best model and generate forecasts"""
        import streamlit as st

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Step 1: Data Preparation
        status_text.text('🔄 Preparing data...')
        progress_bar.progress(10)

        df_ts = self.prepare_data(df, date_col, target_col)
        data = df_ts[target_col].values

        if len(data) < 20:
            raise ValueError("Need at least 20 data points for forecasting")

        # Step 2: Create sequences
        status_text.text('🔢 Creating training sequences...')
        progress_bar.progress(25)

        seq_length = min(10, len(data) // 4)
        X, y = self.create_sequences(data, seq_length)

        if len(X) < 10:
            raise ValueError("Not enough data for sequence creation")

        # Split data
        split_point = int(len(X) * 0.8)
        X_train, X_test = X[:split_point], X[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]

        # Step 3: Train models
        status_text.text('🤖 Training Linear Regression...')
        progress_bar.progress(40)
        self.train_linear_regression(X_train, y_train, X_test, y_test)

        status_text.text('🌲 Training Random Forest...')
        progress_bar.progress(60)
        self.train_random_forest(X_train, y_train, X_test, y_test)

        status_text.text('🧠 Training Neural Network...')
        progress_bar.progress(80)
        self.train_neural_network(X_train, y_train, X_test, y_test)

        # Step 4: Evaluate models
        status_text.text('📊 Evaluating models...')
        progress_bar.progress(90)

        results = self.evaluate_models(y_train, y_test)

        # Find best model based on test RMSE
        best_model_name = min(results.keys(),
                            key=lambda x: results[x]['test_metrics']['RMSE'])

        # Step 5: Generate forecast
        status_text.text('🔮 Generating forecast...')
        progress_bar.progress(95)

        last_sequence = data[-seq_length:]
        forecast = self.forecast_future(best_model_name, forecast_steps, last_sequence)

        # Create best config for compatibility
        best_config = {
            'model_name': best_model_name,
            'model': self.models[best_model_name],
            'metrics': results[best_model_name]['test_metrics'],
            'seq_len': seq_length,
            'model_type': 'ML'
        }

        status_text.text('✅ Forecasting complete!')
        progress_bar.progress(100)

        # Clean up progress indicators
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()

        return best_config, forecast, df_ts

