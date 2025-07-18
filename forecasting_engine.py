import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import xgboost as xgb
import pmdarima as pm
import warnings
warnings.filterwarnings('ignore')

class ForecastingEngine:
    """Main forecasting engine supporting multiple algorithms"""

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

    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        model = xgb.XGBRegressor(random_state=42)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        self.models['XGBoost'] = model
        self.predictions['XGBoost'] = {
            'train': train_pred,
            'test': test_pred
        }

        return model, train_pred, test_pred

    def train_neural_network(self, X_train, y_train, X_test, y_test, hidden_layer_sizes=(100, 50)):
        """Train Neural Network model using scikit-learn MLPRegressor"""
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
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

    def train_arima(self, data, test_size=0.2):
        """Train ARIMA model"""
        split_point = int(len(data) * (1 - test_size))
        train_data = data[:split_point]
        test_data = data[split_point:]

        # Auto ARIMA to find best parameters
        model = pm.auto_arima(train_data,
                             start_p=0, start_q=0,
                             max_p=5, max_q=5,
                             seasonal=False,
                             stepwise=True,
                             suppress_warnings=True)

        # Make predictions
        train_pred = model.fittedvalues()
        test_pred = model.predict(n_periods=len(test_data))

        self.models['ARIMA'] = model
        self.predictions['ARIMA'] = {
            'train': train_pred,
            'test': test_pred
        }

        return model, train_pred, test_pred

    def train_prophet(self, df, date_col, target_col, test_size=0.2):
        """Train Prophet model"""
        # Prepare data for Prophet
        prophet_df = df[[date_col, target_col]].copy()
        prophet_df.columns = ['ds', 'y']

        split_point = int(len(prophet_df) * (1 - test_size))
        train_df = prophet_df[:split_point]
        test_df = prophet_df[split_point:]

        # Train Prophet
        model = Prophet()
        model.fit(train_df)

        # Make predictions
        train_forecast = model.predict(train_df)
        test_forecast = model.predict(test_df)

        self.models['Prophet'] = model
        self.predictions['Prophet'] = {
            'train': train_forecast['yhat'].values,
            'test': test_forecast['yhat'].values
        }

        return model, train_forecast['yhat'].values, test_forecast['yhat'].values

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

        if model_name == 'ARIMA':
            future_pred = model.predict(n_periods=steps)
        elif model_name == 'Prophet':
            future_dates = pd.date_range(start=pd.Timestamp.now(), periods=steps, freq='D')
            future_df = pd.DataFrame({'ds': future_dates})
            future_pred = model.predict(future_df)['yhat'].values
        elif model_name == 'Neural Network':
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
            # For ML models, we need feature engineering for future dates
            # This is a simplified approach - in practice, you'd need proper feature engineering
            raise NotImplementedError(f"Future forecasting for {model_name} requires additional feature engineering")

        return future_pred
