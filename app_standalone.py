import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="COGNOS 2.0 - Time Series Forecasting",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/cognos-2.0',
        'Report a bug': 'https://github.com/yourusername/cognos-2.0/issues',
        'About': """
        # COGNOS 2.0
        Advanced Time Series Forecasting Engine
        
        Built with ❤️ using Streamlit and scikit-learn
        """
    }
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class DataProcessor:
    """Handle data loading, cleaning, and preprocessing for time series forecasting"""

    def __init__(self):
        self.scaler = None

    def detect_date_column(self, df):
        """Automatically detect date/datetime columns"""
        date_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].head())
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

    def create_sequences(self, data, seq_length):
        """Create sequences for ML training"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    def split_data(self, X, y, test_size=0.2):
        """Split data into train and test sets"""
        return train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)

class SimpleForecastingEngine:
    """Simplified forecasting engine with core ML models only"""

    def __init__(self):
        self.models = {}
        self.predictions = {}

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

def main():
    st.markdown('<h1 class="main-header">🧠 COGNOS 2.0</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 2rem;">Advanced Time Series Forecasting Engine</p>', unsafe_allow_html=True)

    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'df_ts' not in st.session_state:
        st.session_state.df_ts = None
    if 'date_col' not in st.session_state:
        st.session_state.date_col = None
    if 'target_col' not in st.session_state:
        st.session_state.target_col = None
    if 'processor' not in st.session_state:
        st.session_state.processor = DataProcessor()
    if 'engine' not in st.session_state:
        st.session_state.engine = SimpleForecastingEngine()

    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["Data Upload & Exploration", "Model Training", "Forecasting", "Model Comparison"]
        )

        st.markdown("---")
        st.header("Quick Actions")
        if st.button("🔄 Reset All"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

        st.markdown("---")
        st.info("📌 Standalone Version: No external dependencies!")

    # Main content based on selected page
    if page == "Data Upload & Exploration":
        data_upload_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Forecasting":
        forecasting_page()
    elif page == "Model Comparison":
        model_comparison_page()

def data_upload_page():
    st.header("📁 Data Upload & Exploration")

    # Data source selection
    data_source = st.radio(
        "Choose data source:",
        ["Upload File", "Sample Data"]
    )

    if data_source == "Upload File":
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv']
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.success("✅ Data loaded successfully!")

            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

    elif data_source == "Sample Data":
        sample_type = st.selectbox(
            "Choose sample data:",
            ["Sine Wave", "Random Walk", "Trend + Seasonality"]
        )

        if st.button("Generate Sample Data"):
            df = generate_sample_data(sample_type)
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.success("✅ Sample data generated successfully!")

    # Display data if loaded
    if st.session_state.data_loaded and st.session_state.df is not None:
        display_data_exploration(st.session_state.df)

def display_data_exploration(df):
    st.subheader("📊 Data Overview")

    # Basic info
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    # Data preview
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    # Column selection for time series
    st.subheader("⚙️ Configure Time Series")

    col1, col2 = st.columns(2)

    with col1:
        # Auto-detect date columns
        date_columns = st.session_state.processor.detect_date_column(df)
        if date_columns:
            date_col = st.selectbox("Select Date Column", date_columns)
        else:
            date_col = st.selectbox("Select Date Column", df.columns)

    with col2:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        target_col = st.selectbox("Select Target Column", numeric_columns)

    if st.button("📈 Visualize Time Series"):
        try:
            # Prepare data
            df_ts = st.session_state.processor.prepare_time_series(df, date_col, target_col)

            # Create interactive plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_ts.index,
                y=df_ts[target_col],
                mode='lines',
                name=target_col,
                line=dict(color='#1f77b4', width=2)
            ))

            fig.update_layout(
                title=f"Time Series: {target_col}",
                xaxis_title="Date",
                yaxis_title=target_col,
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Store configuration
            st.session_state.date_col = date_col
            st.session_state.target_col = target_col
            st.session_state.df_ts = df_ts

        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")

def model_training_page():
    st.header("🤖 Model Training")

    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first in the 'Data Upload & Exploration' page.")
        return

    if st.session_state.df_ts is None:
        st.warning("⚠️ Please configure your time series first by going to 'Data Upload & Exploration' and clicking '📈 Visualize Time Series'.")
        return

    # Model selection
    st.subheader("🎯 Select Models to Train")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Available ML Models:**")
        train_lr = st.checkbox("Linear Regression")
        train_rf = st.checkbox("Random Forest")
        train_nn = st.checkbox("Neural Network (MLP)")

    with col2:
        st.info("💡 **Standalone Version**\n\nThis version includes 3 robust ML algorithms that work reliably without external dependencies.")

    # Training parameters
    st.subheader("⚙️ Training Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
        sequence_length = st.number_input("Sequence Length", 5, 50, 10)

    with col2:
        nn_max_iter = st.number_input("Neural Network Max Iterations", 100, 1000, 300)
        rf_estimators = st.number_input("Random Forest Trees", 50, 500, 100)

    with col3:
        scaling_method = st.selectbox("Scaling Method", ["minmax", "standard"])

    # Train models
    if st.button("🚀 Train Selected Models"):
        if not any([train_lr, train_rf, train_nn]):
            st.error("Please select at least one model to train.")
            return

        with st.spinner("Training models..."):
            try:
                train_models(
                    train_lr, train_rf, train_nn,
                    test_size, sequence_length, nn_max_iter, rf_estimators, scaling_method
                )
                st.session_state.models_trained = True
                st.success("✅ Models trained successfully!")

            except Exception as e:
                st.error(f"Error training models: {str(e)}")
                st.write("Debug info:", str(e))

    # Display training results
    if st.session_state.models_trained:
        display_training_results()

def train_models(train_lr, train_rf, train_nn, test_size, sequence_length, nn_max_iter, rf_estimators, scaling_method):
    try:
        df_ts = st.session_state.df_ts
        target_col = st.session_state.target_col
        engine = st.session_state.engine
        processor = st.session_state.processor

        # Prepare data
        data = df_ts[target_col].values

        # Create lag features
        X, y = processor.create_sequences(data, sequence_length)

        if len(X) == 0:
            raise ValueError("Not enough data to create sequences. Try reducing sequence length.")

        # Split data
        X_train, X_test, y_train, y_test = processor.split_data(X, y, test_size)

        # Train models
        if train_lr:
            engine.train_linear_regression(X_train, y_train, X_test, y_test)

        if train_rf:
            engine.train_random_forest(X_train, y_train, X_test, y_test, rf_estimators)

        if train_nn:
            # Scale data for Neural Network
            scaled_data = processor.scale_data(pd.Series(data), scaling_method)
            X_scaled, y_scaled = processor.create_sequences(scaled_data.flatten(), sequence_length)
            X_train_nn, X_test_nn, y_train_nn, y_test_nn = processor.split_data(X_scaled, y_scaled, test_size)
            engine.train_neural_network(X_train_nn, y_train_nn, X_test_nn, y_test_nn, nn_max_iter)

        # Store split data for evaluation
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

    except Exception as e:
        st.error(f"Training failed: {str(e)}")
        raise e

def display_training_results():
    st.subheader("📈 Training Results")

    # Calculate metrics for all models
    if hasattr(st.session_state, 'y_train') and hasattr(st.session_state, 'y_test'):
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test

        # Display metrics table
        metrics_df = []
        for model_name, predictions in st.session_state.engine.predictions.items():
            train_pred = predictions['train']
            test_pred = predictions['test']

            # Adjust lengths if necessary
            min_train_len = min(len(train_pred), len(y_train))
            min_test_len = min(len(test_pred), len(y_test))

            train_metrics = st.session_state.engine.calculate_metrics(
                y_train[:min_train_len], train_pred[:min_train_len]
            )
            test_metrics = st.session_state.engine.calculate_metrics(
                y_test[:min_test_len], test_pred[:min_test_len]
            )

            row = {
                'Model': model_name,
                'Train MAE': f"{train_metrics['MAE']:.4f}",
                'Train RMSE': f"{train_metrics['RMSE']:.4f}",
                'Train R²': f"{train_metrics['R²']:.4f}",
                'Test MAE': f"{test_metrics['MAE']:.4f}",
                'Test RMSE': f"{test_metrics['RMSE']:.4f}",
                'Test R²': f"{test_metrics['R²']:.4f}"
            }
            metrics_df.append(row)

        metrics_df = pd.DataFrame(metrics_df)
        st.dataframe(metrics_df, use_container_width=True)

def forecasting_page():
    st.header("🔮 Forecasting")

    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first in the 'Model Training' page.")
        return

    # Model selection for forecasting
    available_models = list(st.session_state.engine.models.keys())
    selected_model = st.selectbox("Select Model for Forecasting", available_models)

    # Forecasting parameters
    col1, col2 = st.columns(2)

    with col1:
        forecast_steps = st.number_input("Forecast Steps", 1, 365, 30)

    with col2:
        st.info("💡 Future predictions based on learned patterns")

    if st.button("🔮 Generate Forecast"):
        try:
            with st.spinner("Generating forecast..."):
                # Get last sequence for forecasting
                last_sequence = None
                if hasattr(st.session_state, 'X_test'):
                    last_sequence = st.session_state.X_test[-1]

                forecast = st.session_state.engine.forecast_future(
                    selected_model, forecast_steps, last_sequence
                )

                # Create forecast visualization
                create_forecast_plot(forecast, selected_model)

        except Exception as e:
            st.error(f"Error generating forecast: {str(e)}")

def create_forecast_plot(forecast, model_name):
    # Get historical data
    df_ts = st.session_state.df_ts
    target_col = st.session_state.target_col

    # Create future dates
    last_date = df_ts.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(forecast), freq='D')

    # Create plot
    fig = go.Figure()

    # Historical data (show last 100 points for clarity)
    recent_data = df_ts.tail(100)
    fig.add_trace(go.Scatter(
        x=recent_data.index,
        y=recent_data[target_col],
        mode='lines',
        name='Historical',
        line=dict(color='#1f77b4', width=2)
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=forecast,
        mode='lines',
        name=f'Forecast ({model_name})',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))

    fig.update_layout(
        title=f"Forecast using {model_name}",
        xaxis_title="Date",
        yaxis_title=target_col,
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display forecast table
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast': forecast
    })

    st.subheader("📊 Forecast Data")
    st.dataframe(forecast_df, use_container_width=True)

def model_comparison_page():
    st.header("⚖️ Model Comparison")

    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first in the 'Model Training' page.")
        return

    # Get predictions from all models
    predictions = st.session_state.engine.predictions

    if not predictions:
        st.warning("No predictions available for comparison.")
        return

    # Create comparison visualization
    if hasattr(st.session_state, 'y_test'):
        y_test = st.session_state.y_test

        fig = go.Figure()

        # Plot actual values
        fig.add_trace(go.Scatter(
            x=list(range(len(y_test))),
            y=y_test,
            mode='lines',
            name='Actual',
            line=dict(color='black', width=3)
        ))

        # Plot predictions from each model
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        for i, (model_name, preds) in enumerate(predictions.items()):
            test_pred = preds['test']
            color = colors[i % len(colors)]

            # Adjust length if necessary
            min_len = min(len(test_pred), len(y_test))

            fig.add_trace(go.Scatter(
                x=list(range(min_len)),
                y=test_pred[:min_len],
                mode='lines',
                name=f'{model_name}',
                line=dict(color=color, width=2)
            ))

        fig.update_layout(
            title="Model Predictions vs Actual Values",
            xaxis_title="Time Steps",
            yaxis_title="Values",
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

def generate_sample_data(sample_type):
    """Generate sample time series data"""
    dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')
    n = len(dates)

    if sample_type == "Sine Wave":
        values = 100 + 20 * np.sin(2 * np.pi * np.arange(n) / 365) + np.random.normal(0, 5, n)
    elif sample_type == "Random Walk":
        values = np.cumsum(np.random.normal(0, 1, n)) + 100
    else:  # Trend + Seasonality
        trend = np.linspace(100, 200, n)
        seasonal = 20 * np.sin(2 * np.pi * np.arange(n) / 365)
        noise = np.random.normal(0, 5, n)
        values = trend + seasonal + noise

    return pd.DataFrame({
        'Date': dates,
        'Value': values
    })

if __name__ == "__main__":
    main()
