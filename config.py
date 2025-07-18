# Configuration file for Time Series Forecasting Engine

# Default model parameters
MODEL_CONFIGS = {
    'ARIMA': {
        'max_p': 5,
        'max_q': 5,
        'max_d': 2,
        'seasonal': False,
        'stepwise': True
    },
    'Prophet': {
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10.0,
        'holidays_prior_scale': 10.0,
        'seasonality_mode': 'additive'
    },
    'LSTM': {
        'units': [50, 50],
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 50
    },
    'RandomForest': {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'random_state': 42
    },
    'XGBoost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    }
}

# Data processing settings
DATA_CONFIGS = {
    'test_size': 0.2,
    'sequence_length': 10,
    'scaling_method': 'minmax',
    'freq': 'D'  # Daily frequency
}

# Visualization settings
VIZ_CONFIGS = {
    'theme': 'plotly_white',
    'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
    'figure_height': 500,
    'figure_width': 800
}

# Streamlit app settings
APP_CONFIGS = {
    'page_title': 'Time Series Forecasting Engine',
    'page_icon': '📈',
    'layout': 'wide',
    'sidebar_state': 'expanded'
}
