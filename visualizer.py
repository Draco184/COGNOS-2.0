import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

class Visualizer:
    """Handle all visualization tasks for time series analysis"""

    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    def plot_time_series(self, df, date_col, value_col, title="Time Series Plot"):
        """Create an interactive time series plot"""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df[date_col],
            y=df[value_col],
            mode='lines',
            name=value_col,
            line=dict(color=self.colors[0], width=2)
        ))

        fig.update_layout(
            title=title,
            xaxis_title=date_col,
            yaxis_title=value_col,
            hovermode='x unified',
            template='plotly_white'
        )

        return fig

    def plot_decomposition(self, df, value_col):
        """Plot time series decomposition"""
        from statsmodels.tsa.seasonal import seasonal_decompose

        # Perform decomposition
        decomposition = seasonal_decompose(df[value_col], model='additive', period=365)

        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual'],
            vertical_spacing=0.08
        )

        # Original
        fig.add_trace(go.Scatter(
            x=df.index, y=df[value_col],
            mode='lines', name='Original',
            line=dict(color=self.colors[0])
        ), row=1, col=1)

        # Trend
        fig.add_trace(go.Scatter(
            x=df.index, y=decomposition.trend,
            mode='lines', name='Trend',
            line=dict(color=self.colors[1])
        ), row=2, col=1)

        # Seasonal
        fig.add_trace(go.Scatter(
            x=df.index, y=decomposition.seasonal,
            mode='lines', name='Seasonal',
            line=dict(color=self.colors[2])
        ), row=3, col=1)

        # Residual
        fig.add_trace(go.Scatter(
            x=df.index, y=decomposition.resid,
            mode='lines', name='Residual',
            line=dict(color=self.colors[3])
        ), row=4, col=1)

        fig.update_layout(
            height=800,
            title="Time Series Decomposition",
            showlegend=False
        )

        return fig

    def plot_forecast_comparison(self, actual, predictions, model_names, dates=None):
        """Compare forecasts from multiple models"""
        fig = go.Figure()

        # Plot actual values
        if dates is not None:
            x_values = dates
        else:
            x_values = list(range(len(actual)))

        fig.add_trace(go.Scatter(
            x=x_values,
            y=actual,
            mode='lines',
            name='Actual',
            line=dict(color='black', width=3)
        ))

        # Plot predictions from each model
        for i, (model_name, pred) in enumerate(zip(model_names, predictions)):
            fig.add_trace(go.Scatter(
                x=x_values[:len(pred)],
                y=pred,
                mode='lines',
                name=model_name,
                line=dict(color=self.colors[i % len(self.colors)], width=2)
            ))

        fig.update_layout(
            title="Model Predictions Comparison",
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified',
            template='plotly_white'
        )

        return fig

    def plot_residuals(self, actual, predicted, model_name):
        """Plot residual analysis"""
        residuals = actual - predicted

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Residuals vs Time',
                'Residuals vs Fitted',
                'Residual Distribution',
                'Q-Q Plot'
            ]
        )

        # Residuals vs Time
        fig.add_trace(go.Scatter(
            x=list(range(len(residuals))),
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(color=self.colors[0])
        ), row=1, col=1)

        # Residuals vs Fitted
        fig.add_trace(go.Scatter(
            x=predicted,
            y=residuals,
            mode='markers',
            name='Residuals vs Fitted',
            marker=dict(color=self.colors[1])
        ), row=1, col=2)

        # Residual Distribution
        fig.add_trace(go.Histogram(
            x=residuals,
            name='Distribution',
            marker=dict(color=self.colors[2])
        ), row=2, col=1)

        # Q-Q Plot
        from scipy import stats
        sorted_residuals = np.sort(residuals)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))

        fig.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=sorted_residuals,
            mode='markers',
            name='Q-Q Plot',
            marker=dict(color=self.colors[3])
        ), row=2, col=2)

        fig.update_layout(
            height=600,
            title=f"Residual Analysis - {model_name}",
            showlegend=False
        )

        return fig

    def plot_feature_importance(self, model, feature_names):
        """Plot feature importance for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_

            fig = go.Figure(go.Bar(
                x=importances,
                y=feature_names,
                orientation='h',
                marker=dict(color=self.colors[0])
            ))

            fig.update_layout(
                title="Feature Importance",
                xaxis_title="Importance",
                yaxis_title="Features",
                template='plotly_white'
            )

            return fig
        else:
            return None

    def plot_correlation_matrix(self, df):
        """Plot correlation matrix heatmap"""
        corr = df.select_dtypes(include=[np.number]).corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmid=0
        ))

        fig.update_layout(
            title="Correlation Matrix",
            template='plotly_white'
        )

        return fig
