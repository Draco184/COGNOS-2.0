import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from data_processor import DataProcessor
from simple_forecasting_engine import SimpleForecastingEngine
from datetime import datetime, timedelta
import io

# Page configuration
st.set_page_config(
    page_title="Time Series Forecasting Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
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

def main():
    st.markdown('<h1 class="main-header">📈 Time Series Forecasting Engine (Core Version)</h1>', unsafe_allow_html=True)

    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'df' not in st.session_state:
        st.session_state.df = None
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
        st.info("📌 Core Version: Linear Regression, Random Forest, Neural Network")

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

    # Model selection
    st.subheader("🎯 Select Models to Train")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Available ML Models:**")
        train_lr = st.checkbox("Linear Regression")
        train_rf = st.checkbox("Random Forest")
        train_nn = st.checkbox("Neural Network (MLP)")

    with col2:
        st.info("💡 **Core Version Active**\n\nThis version includes 3 robust ML algorithms that work reliably across all systems.")

    # Training parameters
    st.subheader("⚙️ Training Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
        sequence_length = st.number_input("Sequence Length (for ML models)", 5, 50, 10)

    with col2:
        nn_max_iter = st.number_input("Neural Network Max Iterations", 100, 1000, 500)
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

    # Display training results
    if st.session_state.models_trained:
        display_training_results()

def train_models(train_lr, train_rf, train_nn, test_size, sequence_length, nn_max_iter, rf_estimators, scaling_method):
    df_ts = st.session_state.df_ts
    target_col = st.session_state.target_col
    engine = st.session_state.engine
    processor = st.session_state.processor

    # Prepare data for different model types
    data = df_ts[target_col].values

    # Create lag features
    X, y = processor.create_sequences(data, sequence_length)

    if train_nn:
        # Scale data for Neural Network
        scaled_data = processor.scale_data(pd.Series(data), scaling_method)
        X_scaled, y_scaled = processor.create_sequences(scaled_data.flatten(), sequence_length)
        X_train_nn, X_test_nn, y_train_nn, y_test_nn = processor.split_data(
            X_scaled, y_scaled, test_size
        )

    # Split for other ML models
    X_train, X_test, y_train, y_test = processor.split_data(X, y, test_size)

    # Train ML models
    if train_lr:
        engine.train_linear_regression(X_train, y_train, X_test, y_test)

    if train_rf:
        engine.train_random_forest(X_train, y_train, X_test, y_test, rf_estimators)

    if train_nn:
        engine.train_neural_network(X_train_nn, y_train_nn, X_test_nn, y_test_nn, nn_max_iter)

    # Store split data for evaluation
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test

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

    # Historical data
    fig.add_trace(go.Scatter(
        x=df_ts.index,
        y=df_ts[target_col],
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
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

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
