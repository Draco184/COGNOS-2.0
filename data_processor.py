import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Handle data loading, cleaning, and preprocessing for time series forecasting"""

    def __init__(self):
        self.scaler = None
        self.original_columns = None

    def load_data(self, file_path=None, data=None):
        """Load data from file or accept DataFrame directly"""
        if file_path:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel files.")
        elif data is not None:
            df = data.copy()
        else:
            raise ValueError("Either file_path or data must be provided")

        return df

    def detect_date_column(self, df):
        """Automatically detect date/datetime columns"""
        date_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col])
                    date_columns.append(col)
                except:
                    continue
        return date_columns

    def prepare_time_series(self, df, date_col, target_col, freq='D'):
        """Prepare data for time series analysis"""
        # Convert date column to datetime
        df[date_col] = pd.to_datetime(df[date_col])

        # Set date as index
        df = df.set_index(date_col)

        # Sort by date
        df = df.sort_index()

        # Resample if needed
        if freq:
            df = df.resample(freq).mean()

        # Remove missing values
        df = df.dropna()

        return df

    def scale_data(self, data, method='minmax'):
        """Scale the data for neural networks"""
        if method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'standard':
            self.scaler = StandardScaler()

        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
        return scaled_data

    def inverse_scale(self, scaled_data):
        """Inverse transform scaled data"""
        return self.scaler.inverse_transform(scaled_data)

    def create_sequences(self, data, seq_length):
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=False)
