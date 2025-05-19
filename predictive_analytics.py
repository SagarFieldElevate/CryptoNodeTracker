"""
Predictive analytics module for blockchain metrics
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def predict_network_metrics(historical_data, days_to_predict=7):
    """
    Generate predictions for network metrics based on historical data
    
    Parameters:
    - historical_data: DataFrame with timestamp index and columns for metrics
    - days_to_predict: Number of days to forecast
    
    Returns:
    - DataFrame with predictions
    """
    try:
        predictions = {}
        metrics = historical_data.columns
        
        for metric in metrics:
            try:
                # Get time series data for this metric
                metric_data = historical_data[metric].dropna()
                
                if len(metric_data) < 5:  # Need enough data for meaningful prediction
                    logging.warning(f"Insufficient data for predicting {metric}")
                    continue
                
                # Try ARIMA model for time series forecasting
                predictions[metric] = forecast_with_arima(metric_data, days_to_predict)
                
            except Exception as e:
                logging.error(f"Error predicting {metric}: {str(e)}")
                # Fallback to simpler model
                try:
                    predictions[metric] = forecast_with_exponential_smoothing(metric_data, days_to_predict)
                except:
                    # Last resort: linear regression
                    predictions[metric] = forecast_with_linear_regression(metric_data, days_to_predict)
        
        # Create prediction DataFrame
        last_date = historical_data.index[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(days_to_predict)]
        
        pred_df = pd.DataFrame(index=future_dates)
        for metric, values in predictions.items():
            pred_df[metric] = values
            
        return pred_df
    
    except Exception as e:
        logging.error(f"Error in predict_network_metrics: {str(e)}")
        return pd.DataFrame()

def forecast_with_arima(series, days_to_predict):
    """Forecast using ARIMA model"""
    # Convert to numpy array
    data = np.array(series)
    
    # Simple order selection for ARIMA
    order = (1, 1, 1)
    
    # Fit model
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    
    # Make prediction
    forecast = model_fit.forecast(steps=days_to_predict)
    return forecast

def forecast_with_exponential_smoothing(series, days_to_predict):
    """Forecast using Exponential Smoothing"""
    # Convert to numpy array
    data = np.array(series)
    
    # Fit model with trend
    model = ExponentialSmoothing(data, trend='add', seasonal=None)
    model_fit = model.fit()
    
    # Make prediction
    forecast = model_fit.forecast(days_to_predict)
    return forecast

def forecast_with_linear_regression(series, days_to_predict):
    """Simple forecast using linear regression"""
    # Create features (X) as the index of points
    X = np.array(range(len(series))).reshape(-1, 1)
    y = np.array(series)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit model
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # Predict future points
    future_X = np.array(range(len(series), len(series) + days_to_predict)).reshape(-1, 1)
    future_X_scaled = scaler.transform(future_X)
    forecast = model.predict(future_X_scaled)
    
    return forecast

def calculate_trend_confidence(historical_data, predictions):
    """
    Calculate confidence in trend predictions
    
    Parameters:
    - historical_data: DataFrame with historical values
    - predictions: DataFrame with predicted values
    
    Returns:
    - Dictionary with confidence scores for each metric
    """
    confidence = {}
    
    for metric in predictions.columns:
        if metric in historical_data.columns:
            # Use the last part of historical data to validate model performance
            train_size = int(len(historical_data) * 0.8)
            train_data = historical_data[metric].iloc[:train_size]
            test_data = historical_data[metric].iloc[train_size:]
            
            if len(test_data) >= 3:  # Need enough test data
                # Train model on training data
                try:
                    model = ARIMA(train_data, order=(1, 1, 1))
                    model_fit = model.fit()
                    
                    # Predict for test period
                    preds = model_fit.forecast(steps=len(test_data))
                    
                    # Calculate RMSE
                    rmse = np.sqrt(mean_squared_error(test_data, preds))
                    
                    # Calculate confidence (inverse of normalized RMSE)
                    # Higher error = lower confidence
                    data_range = historical_data[metric].max() - historical_data[metric].min()
                    if data_range > 0:
                        normalized_rmse = rmse / data_range
                        confidence_score = max(0, 1 - normalized_rmse)
                        confidence[metric] = min(confidence_score, 0.95)  # Cap at 95%
                    else:
                        confidence[metric] = 0.5  # Neutral if no range
                except:
                    confidence[metric] = 0.5  # Default to neutral confidence
            else:
                confidence[metric] = 0.4  # Lower confidence with limited test data
        else:
            confidence[metric] = 0.3  # Low confidence for metrics without historical data
    
    return confidence

def detect_anomalies(historical_data, window_size=5, threshold=2.0):
    """
    Detect anomalies in blockchain metrics using rolling statistics
    
    Parameters:
    - historical_data: DataFrame with blockchain metrics
    - window_size: Size of the rolling window
    - threshold: Number of standard deviations to consider as anomaly
    
    Returns:
    - DataFrame with detected anomalies
    """
    anomalies = {}
    
    for column in historical_data.columns:
        series = historical_data[column].dropna()
        
        if len(series) <= window_size:
            continue
            
        # Calculate rolling mean and standard deviation
        rolling_mean = series.rolling(window=window_size).mean()
        rolling_std = series.rolling(window=window_size).std()
        
        # Calculate z-scores
        z_scores = (series - rolling_mean) / rolling_std
        
        # Flag anomalies where z-score exceeds threshold
        anomalies[column] = series[abs(z_scores) > threshold]
    
    return anomalies

def get_predictive_indicators(historical_data, days_to_predict=7):
    """
    Generate comprehensive predictive indicators from historical blockchain data
    
    Parameters:
    - historical_data: Dictionary with different metrics over time
    - days_to_predict: Number of days to forecast
    
    Returns:
    - Dictionary with predictions, trends, confidence scores and anomalies
    """
    try:
        # Convert dictionary of metrics to DataFrame
        df_metrics = {}
        
        # Process network metrics data
        if 'network_metrics_history' in historical_data:
            df_network = pd.DataFrame(historical_data['network_metrics_history'])
            if not df_network.empty:
                df_metrics['network'] = df_network
                
        # Process address metrics data
        if 'address_metrics_history' in historical_data:
            df_address = pd.DataFrame(historical_data['address_metrics_history'])
            if not df_address.empty:
                df_metrics['address'] = df_address
        
        # Process any other metrics categories
        
        results = {
            'predictions': {},
            'confidence': {},
            'anomalies': {},
            'trends': {}
        }
        
        # Generate predictions for each metrics category
        for category, df in df_metrics.items():
            if not df.empty and len(df) >= 5:  # Need minimum data points
                predictions = predict_network_metrics(df, days_to_predict)
                results['predictions'][category] = predictions
                
                # Calculate confidence in predictions
                confidence = calculate_trend_confidence(df, predictions)
                results['confidence'][category] = confidence
                
                # Detect anomalies
                anomalies = detect_anomalies(df)
                results['anomalies'][category] = anomalies
                
                # Determine trends (increasing, decreasing, stable)
                trends = {}
                for col in df.columns:
                    if len(df[col].dropna()) > 3:
                        # Calculate trend direction as slope of linear regression
                        x = np.array(range(len(df[col].dropna()))).reshape(-1, 1)
                        y = df[col].dropna().values
                        model = LinearRegression()
                        model.fit(x, y)
                        slope = model.coef_[0]
                        
                        # Classify trend based on slope and data magnitude
                        magnitude = df[col].mean()
                        if abs(slope) < 0.01 * magnitude:
                            trends[col] = "Stable"
                        elif slope > 0:
                            if slope > 0.05 * magnitude:
                                trends[col] = "Strongly Increasing"
                            else:
                                trends[col] = "Moderately Increasing"
                        else:
                            if abs(slope) > 0.05 * magnitude:
                                trends[col] = "Strongly Decreasing"
                            else:
                                trends[col] = "Moderately Decreasing"
                                
                results['trends'][category] = trends
                
        return results
    
    except Exception as e:
        logging.error(f"Error in get_predictive_indicators: {str(e)}")
        return {
            'error': str(e),
            'message': 'Failed to generate predictive indicators'
        }