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
