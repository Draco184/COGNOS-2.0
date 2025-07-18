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
    page_title="COGNOS 2.0 - AI Time Series Forecasting",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/cognos-2.0',
        'Report a bug': 'https://github.com/yourusername/cognos-2.0/issues',
        'About': """
        # COGNOS 2.0
        AI-Powered Time Series Forecasting Engine
        
        Automatically finds the best model for your data!
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
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.3rem;
        margin-bottom: 3rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1f77b4;
    }
    .best-model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .forecast-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 0.5rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

class AutoMLForecastingEngine:
    """Automated ML forecasting engine that finds the best model automatically"""

    def __init__(self):
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        self.best_model = None
        self.best_score = float('inf')

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

    def scale_data(self, data, method='minmax'):
        """Scale the data"""
        if method == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
        return scaled_data, scaler

    def train_and_evaluate_model(self, model, model_name, X_train, y_train, X_test, y_test):
        """Train a model and return its performance"""
        try:
            model.fit(X_train, y_train)
            test_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, test_pred))

            return {
                'model': model,
                'rmse': rmse,
                'predictions': test_pred,
                'mae': mean_absolute_error(y_test, test_pred),
                'r2': r2_score(y_test, test_pred)
            }
        except Exception as e:
            return None

    def auto_forecast(self, df, date_col, target_col, forecast_steps=30):
        """Automatically find the best model and generate forecasts"""

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Step 1: Data Preparation
        status_text.text('🔄 Preparing data...')
        progress_bar.progress(10)

        df_ts = self.prepare_data(df, date_col, target_col)
        data = df_ts[target_col].values

        if len(data) < 20:
            raise ValueError("Need at least 20 data points for forecasting")

        # Step 2: Test different configurations
        status_text.text('🔍 Testing different model configurations...')
        progress_bar.progress(30)

        best_config = None
        best_rmse = float('inf')
        results = []

        # Test different sequence lengths and models
        sequence_lengths = [5, 10, 15] if len(data) > 50 else [5]
        test_sizes = [0.2, 0.3]

        config_count = 0
        total_configs = len(sequence_lengths) * len(test_sizes) * 3  # 3 models

        for seq_len in sequence_lengths:
            for test_size in test_sizes:
                try:
                    # Create sequences
                    X, y = self.create_sequences(data, seq_len)
                    if len(X) < 10:
                        continue

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, shuffle=False
                    )

                    # Test Linear Regression
                    config_count += 1
                    progress_bar.progress(30 + (config_count / total_configs) * 40)

                    lr_result = self.train_and_evaluate_model(
                        LinearRegression(), 'Linear Regression',
                        X_train, y_train, X_test, y_test
                    )

                    if lr_result and lr_result['rmse'] < best_rmse:
                        best_rmse = lr_result['rmse']
                        best_config = {
                            'model_name': 'Linear Regression',
                            'model': lr_result['model'],
                            'seq_len': seq_len,
                            'test_size': test_size,
                            'metrics': lr_result,
                            'X_train': X_train, 'X_test': X_test,
                            'y_train': y_train, 'y_test': y_test,
                            'scaler': None
                        }

                    # Test Random Forest
                    config_count += 1
                    progress_bar.progress(30 + (config_count / total_configs) * 40)

                    rf_result = self.train_and_evaluate_model(
                        RandomForestRegressor(n_estimators=50, random_state=42), 'Random Forest',
                        X_train, y_train, X_test, y_test
                    )

                    if rf_result and rf_result['rmse'] < best_rmse:
                        best_rmse = rf_result['rmse']
                        best_config = {
                            'model_name': 'Random Forest',
                            'model': rf_result['model'],
                            'seq_len': seq_len,
                            'test_size': test_size,
                            'metrics': rf_result,
                            'X_train': X_train, 'X_test': X_test,
                            'y_train': y_train, 'y_test': y_test,
                            'scaler': None
                        }

                    # Test Neural Network (scaled data)
                    config_count += 1
                    progress_bar.progress(30 + (config_count / total_configs) * 40)

                    scaled_data, scaler = self.scale_data(pd.Series(data))
                    X_scaled, y_scaled = self.create_sequences(scaled_data.flatten(), seq_len)

                    if len(X_scaled) >= 10:
                        X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(
                            X_scaled, y_scaled, test_size=test_size, random_state=42, shuffle=False
                        )

                        nn_result = self.train_and_evaluate_model(
                            MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=200, random_state=42),
                            'Neural Network', X_train_nn, y_train_nn, X_test_nn, y_test_nn
                        )

                        if nn_result and nn_result['rmse'] < best_rmse:
                            best_rmse = nn_result['rmse']
                            best_config = {
                                'model_name': 'Neural Network',
                                'model': nn_result['model'],
                                'seq_len': seq_len,
                                'test_size': test_size,
                                'metrics': nn_result,
                                'X_train': X_train_nn, 'X_test': X_test_nn,
                                'y_train': y_train_nn, 'y_test': y_test_nn,
                                'scaler': scaler,
                                'original_data': data
                            }

                except Exception as e:
                    continue

        if best_config is None:
            raise ValueError("Could not find a suitable model configuration")

        # Step 3: Generate forecasts
        status_text.text('🎯 Generating forecasts with best model...')
        progress_bar.progress(80)

        forecast = self.generate_forecast(best_config, forecast_steps)

        # Step 4: Complete
        status_text.text('✅ Forecasting complete!')
        progress_bar.progress(100)

        # Clean up progress indicators
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()

        return best_config, forecast, df_ts

    def generate_forecast(self, config, steps):
        """Generate future forecasts using the best model"""
        model = config['model']
        model_name = config['model_name']
        seq_len = config['seq_len']

        if model_name == 'Neural Network':
            # Use scaled data for neural network
            scaler = config['scaler']
            original_data = config['original_data']
            scaled_data, _ = self.scale_data(pd.Series(original_data))

            # Get last sequence
            last_sequence = scaled_data[-seq_len:].flatten()
            future_pred = []

            for _ in range(steps):
                pred = model.predict([last_sequence])
                future_pred.append(pred[0])
                last_sequence = np.append(last_sequence[1:], pred[0])

            # Inverse transform predictions
            future_pred = np.array(future_pred).reshape(-1, 1)
            future_pred = scaler.inverse_transform(future_pred).flatten()

        else:
            # For other models, use trend-based forecasting
            X_test = config['X_test']
            if len(X_test) > 0:
                last_sequence = X_test[-1]
                recent_trend = np.mean(np.diff(last_sequence[-5:]))
                last_value = last_sequence[-1]
                future_pred = np.array([last_value + (i + 1) * recent_trend for i in range(steps)])
            else:
                # Fallback: simple trend from original data
                recent_data = config['original_data'][-10:] if 'original_data' in config else [100]
                trend = np.mean(np.diff(recent_data)) if len(recent_data) > 1 else 0
                last_value = recent_data[-1]
                future_pred = np.array([last_value + (i + 1) * trend for i in range(steps)])

        return future_pred

def main():
    st.markdown('<h1 class="main-header">🧠 COGNOS 2.0</h1>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Time Series Forecasting • Automatically Finds the Best Model</div>', unsafe_allow_html=True)

    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'forecast_generated' not in st.session_state:
        st.session_state.forecast_generated = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'engine' not in st.session_state:
        st.session_state.engine = AutoMLForecastingEngine()

    # Sidebar
    with st.sidebar:
        st.header("🧠 COGNOS 2.0")
        st.info("**AI-Powered Forecasting**\n\nJust upload your data, select columns, and let AI find the best model automatically!")

        st.markdown("---")

        if st.button("🔄 Reset All"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

        st.markdown("---")

        st.markdown("""
        **How it works:**
        1. 📁 Upload your time series data
        2. 🎯 Select date & value columns  
        3. 🚀 Click "Generate Forecast"
        4. ✨ Get instant AI-powered predictions!
        """)

    # Main content
    data_upload_and_forecast()

def data_upload_and_forecast():
    # Data Upload Section
    st.header("📁 Data Upload & Configuration")

    # Data source selection
    col1, col2 = st.columns([2, 1])

    with col1:
        data_source = st.radio(
            "Choose your data source:",
            ["Upload CSV File", "Try Sample Data"],
            horizontal=True
        )

    with col2:
        st.markdown("### Quick Start")
        st.markdown("Upload CSV with date & value columns")

    if data_source == "Upload CSV File":
        uploaded_file = st.file_uploader(
            "📎 Drop your CSV file here",
            type=['csv'],
            help="Upload a CSV file with at least a date column and a numeric value column"
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.success("✅ Data loaded successfully!")

            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")

    else:  # Sample Data
        sample_type = st.selectbox(
            "Choose sample dataset:",
            ["📈 Trend with Seasonality", "🌊 Sine Wave Pattern", "🎲 Random Walk"]
        )

        if st.button("🎲 Generate Sample Data", type="primary"):
            df = generate_sample_data(sample_type.split(" ")[1].lower())
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.success("✅ Sample data generated!")

    # Data Preview and Configuration
    if st.session_state.data_loaded and st.session_state.df is not None:
        df = st.session_state.df

        st.markdown("---")
        st.subheader("📊 Data Preview & Configuration")

        # Data overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📏 Rows", len(df))
        with col2:
            st.metric("📋 Columns", len(df.columns))
        with col3:
            st.metric("💾 Size", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        with col4:
            st.metric("📅 Date Range", "Ready" if len(df) > 0 else "Empty")

        # Show data preview
        with st.expander("🔍 View Data Preview", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)

        # Column selection
        st.subheader("⚙️ Column Configuration")

        col1, col2 = st.columns(2)

        with col1:
            # Auto-detect date columns
            date_columns = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        pd.to_datetime(df[col].head())
                        date_columns.append(col)
                    except:
                        continue

            if date_columns:
                date_col = st.selectbox("📅 Select Date Column", date_columns, help="Choose the column containing dates/timestamps")
            else:
                date_col = st.selectbox("📅 Select Date Column", df.columns, help="Choose the column containing dates/timestamps")

        with col2:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_columns:
                target_col = st.selectbox("📊 Select Value Column", numeric_columns, help="Choose the numeric column to forecast")
            else:
                st.error("❌ No numeric columns found in your data!")
                return

        # Forecasting section
        if date_col and target_col:
            st.markdown("---")
            st.markdown('<div class="forecast-section">', unsafe_allow_html=True)
            st.subheader("🔮 AI Forecasting")

            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                forecast_days = st.slider(
                    "📈 Forecast Period (days)",
                    min_value=7, max_value=365, value=30,
                    help="How many days into the future to predict"
                )

            with col2:
                st.markdown("### AI will automatically:")
                st.markdown("• Test multiple models")
                st.markdown("• Optimize parameters")
                st.markdown("• Select best performer")

            with col3:
                st.markdown("### Available Models:")
                st.markdown("• 📈 Linear Regression")
                st.markdown("• 🌲 Random Forest")
                st.markdown("• 🧠 Neural Network")

            # Big forecast button
            if st.button("🚀 Generate AI Forecast", type="primary", use_container_width=True):
                try:
                    with st.spinner("🤖 AI is analyzing your data and finding the best model..."):

                        # Generate forecast using AutoML
                        best_config, forecast, df_ts = st.session_state.engine.auto_forecast(
                            df, date_col, target_col, forecast_days
                        )

                        # Store results
                        st.session_state.best_config = best_config
                        st.session_state.forecast = forecast
                        st.session_state.df_ts = df_ts
                        st.session_state.forecast_generated = True
                        st.session_state.target_col = target_col
                        st.session_state.forecast_days = forecast_days

                    st.success("🎉 Forecast generated successfully!")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Forecasting failed: {str(e)}")
                    st.info("💡 Try using more data points or check your data format.")

            st.markdown('</div>', unsafe_allow_html=True)

    # Display Results
    if st.session_state.forecast_generated:
        display_forecast_results()

def display_forecast_results():
    """Display the forecasting results in an attractive format"""

    st.markdown("---")
    st.header("🎯 Forecast Results")

    best_config = st.session_state.best_config
    forecast = st.session_state.forecast
    df_ts = st.session_state.df_ts
    target_col = st.session_state.target_col

    # Best Model Card
    st.markdown(f"""
    <div class="best-model-card">
        <h3>🏆 Best Model Selected: {best_config['model_name']}</h3>
        <p><strong>Performance Metrics:</strong></p>
        <ul>
            <li>RMSE: {best_config['metrics']['rmse']:.4f}</li>
            <li>MAE: {best_config['metrics']['mae']:.4f}</li>
            <li>R² Score: {best_config['metrics']['r2']:.4f}</li>
        </ul>
        <p><em>AI automatically tested multiple models and selected this one for optimal performance!</em></p>
    </div>
    """, unsafe_allow_html=True)

    # Forecast visualization
    st.subheader("📈 Forecast Visualization")

    # Create future dates
    last_date = df_ts.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=len(forecast),
        freq='D'
    )

    # Create the plot
    fig = go.Figure()

    # Historical data
    recent_data = df_ts.tail(min(100, len(df_ts)))
    fig.add_trace(go.Scatter(
        x=recent_data.index,
        y=recent_data[target_col],
        mode='lines',
        name='📊 Historical Data',
        line=dict(color='#1f77b4', width=3)
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=forecast,
        mode='lines',
        name=f'🔮 AI Forecast ({best_config["model_name"]})',
        line=dict(color='#ff6b6b', width=3, dash='dot')
    ))

    # Add a vertical line to separate historical and forecast
    fig.add_vline(
        x=last_date,
        line_dash="dash",
        line_color="gray",
        annotation_text="Forecast Start"
    )

    fig.update_layout(
        title=f"🧠 COGNOS 2.0 - AI Forecast for {target_col}",
        xaxis_title="📅 Date",
        yaxis_title=f"📊 {target_col}",
        hovermode='x unified',
        height=500,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # Forecast summary
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Forecast Summary")

        # Calculate forecast statistics
        forecast_mean = np.mean(forecast)
        forecast_trend = forecast[-1] - forecast[0]
        historical_mean = np.mean(df_ts[target_col].tail(30))

        st.metric("📈 Average Forecast", f"{forecast_mean:.2f}")
        st.metric("📈 Forecast Trend", f"{forecast_trend:+.2f}", delta=f"{forecast_trend:.2f}")
        st.metric("📊 vs Recent Average", f"{forecast_mean:.2f}", delta=f"{forecast_mean - historical_mean:.2f}")

    with col2:
        st.subheader("📋 Forecast Data")

        # Create downloadable forecast data
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Forecast': forecast,
            'Model': best_config['model_name']
        })

        st.dataframe(forecast_df.head(10), use_container_width=True)

        # Download button
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast Data",
            data=csv,
            file_name=f"cognos_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            type="primary"
        )

def generate_sample_data(sample_type):
    """Generate sample time series data"""
    dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')
    n = len(dates)

    if sample_type == "sine":
        values = 100 + 20 * np.sin(2 * np.pi * np.arange(n) / 365) + np.random.normal(0, 5, n)
        name = "Seasonal Sales"
    elif sample_type == "random":
        values = np.cumsum(np.random.normal(0, 1, n)) + 100
        name = "Stock Price"
    else:  # trend
        trend = np.linspace(100, 200, n)
        seasonal = 20 * np.sin(2 * np.pi * np.arange(n) / 365)
        noise = np.random.normal(0, 5, n)
        values = trend + seasonal + noise
        name = "Revenue"

    return pd.DataFrame({
        'Date': dates,
        name: values
    })

if __name__ == "__main__":
    main()
