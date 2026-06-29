"""
Gold Futures Price Prediction Module (Version 4)

This module provides a function to predict the next gold futures price
based on historical price data, with strict constraints on downward trends.
"""

import numpy as np


def predict_next_gold_price(prices):
    """
    Predicts the next gold futures price based on historical data.
    
    This function uses a combination of trend analysis, moving averages, and
    exponential smoothing to predict the next price. It enforces a critical
    constraint: if the recent trend is downward, the prediction will never
    exceed the last known price.
    
    Args:
        prices: List or array of up to 100 historical prices (most recent last).
                Must contain at least 1 price.
        
    Returns:
        tuple: (predicted_price, confidence_level)
            - predicted_price: float rounded to 2 decimal places
            - confidence_level: int between 1-100, based on trend consistency
                               and volatility
    
    Raises:
        ValueError: If prices list is empty or invalid
        
    Example:
        >>> prices = [1969.03, 1968.95, 1968.16, 1968.68, 1968.47]
        >>> prediction, confidence = predict_next_gold_price(prices)
        >>> print(f"Predicted: ${prediction:.2f} (Confidence: {confidence}%)")
    """
    if not prices or len(prices) == 0:
        raise ValueError("Price list cannot be empty")
    
    # Convert to numpy array and limit to last 100 points
    prices = np.array(prices[-100:], dtype=float)
    n = len(prices)
    
    # Handle edge case: only one data point
    if n == 1:
        return round(float(prices[0]), 2), 50
    
    last_price = prices[-1]
    
    # Determine window sizes based on available data
    short_window = min(5, n)
    medium_window = min(20, n)
    long_window = min(50, n)
    
    # Extract windows
    recent_prices = prices[-short_window:]
    medium_prices = prices[-medium_window:]
    long_prices = prices[-long_window:]
    
    # 1. LINEAR TREND ANALYSIS
    def calculate_trend(data):
        """Calculate linear trend (slope) using least squares regression."""
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 1)
        return coeffs[0]  # slope
    
    short_trend = calculate_trend(recent_prices)
    medium_trend = calculate_trend(medium_prices) if n >= 3 else short_trend
    
    # 2. WEIGHTED MOVING AVERAGE (exponential weights favor recent prices)
    weights = np.exp(np.linspace(-1, 0, short_window))
    weights /= weights.sum()
    wma = np.sum(recent_prices * weights)
    
    # 3. EXPONENTIAL MOVING AVERAGE
    alpha = 0.3  # Smoothing factor
    ema = prices[0]
    for price in prices[1:]:
        ema = alpha * price + (1 - alpha) * ema
    
    # 4. MOMENTUM-BASED PREDICTION
    # Weight recent trend more heavily than medium-term trend
    trend_component = short_trend * 0.7 + medium_trend * 0.3
    trend_prediction = last_price + trend_component
    
    # 5. COMBINE PREDICTIONS (weighted ensemble)
    prediction = (
        trend_prediction * 0.4 +  # Trend continuation
        wma * 0.3 +                # Weighted moving average
        ema * 0.2 +                # Exponential smoothing
        last_price * 0.1           # Anchor to current price
    )
    
    # 6. APPLY CRITICAL CONSTRAINT
    # If the last price movement was downward, cap prediction at last price
    # This ensures we never predict an increase when the trend is declining
    if n >= 2:
        recent_change = prices[-1] - prices[-2]
        if recent_change < 0:
            # Downward movement detected - don't predict higher than current
            prediction = min(prediction, last_price)
            
        # Additional check: if multiple recent declines, be more conservative
        if n >= 3:
            recent_changes = np.diff(prices[-3:])
            declining_count = np.sum(recent_changes < 0)
            if declining_count >= 2:
                # Strong downward trend - apply dampening
                prediction = min(prediction, last_price - abs(short_trend) * 0.5)
    
    # 7. CALCULATE CONFIDENCE LEVEL
    # Base confidence
    base_confidence = 65
    
    # Factor 1: Volatility (lower volatility = higher confidence)
    if n >= medium_window:
        returns = np.diff(medium_prices) / medium_prices[:-1]
        volatility = np.std(returns)
        # Normalize volatility impact (typical gold volatility ~0.001-0.01)
        volatility_penalty = min(35, volatility * 5000)
    else:
        volatility_penalty = 20  # Default penalty for insufficient data
    
    # Factor 2: Trend consistency (more consistent = higher confidence)
    if n >= short_window:
        price_changes = np.diff(recent_prices)
        # Check if changes are in same direction
        sign_changes = np.diff(np.sign(price_changes))
        consistency_ratio = 1 - (np.count_nonzero(sign_changes) / max(1, len(sign_changes)))
        consistency_bonus = consistency_ratio * 25
    else:
        consistency_bonus = 10
    
    # Factor 3: Data availability bonus
    data_bonus = min(10, (n / 100) * 10)
    
    # Compute final confidence
    confidence = base_confidence - volatility_penalty + consistency_bonus + data_bonus
    confidence = int(np.clip(confidence, 1, 100))
    
    # Return prediction rounded to 2 decimal places
    return round(float(prediction), 2), confidence


def predict_with_analysis(prices):
    """
    Extended prediction function that returns additional analysis details.
    
    Args:
        prices: List or array of historical prices
        
    Returns:
        dict: Contains prediction, confidence, and analysis details
    """
    prediction, confidence = predict_next_gold_price(prices)
    
    prices = np.array(prices[-100:], dtype=float)
    n = len(prices)
    
    # Calculate additional metrics
    if n >= 2:
        last_change = prices[-1] - prices[-2]
        last_change_pct = (last_change / prices[-2]) * 100
    else:
        last_change = 0
        last_change_pct = 0
    
    if n >= 5:
        short_trend_direction = "DOWN" if np.diff(prices[-5:]).mean() < 0 else "UP"
    else:
        short_trend_direction = "NEUTRAL"
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "last_price": round(float(prices[-1]), 2),
        "last_change": round(float(last_change), 2),
        "last_change_pct": round(float(last_change_pct), 3),
        "trend_direction": short_trend_direction,
        "data_points_used": n
    }


if __name__ == "__main__":
    # Test with the provided example sequence
    test_data = [
        1969.03, 1968.95, 1968.16, 1968.68, 1968.47, 1967.82, 1967.62, 1965.59,
        1967.34, 1967.68, 1968.26, 1965.51, 1962.06, 1964.5, 1962.83, 1964.87,
        1960.23, 1955.41, 1954.87, 1953.75, 1951.99, 1948.2, 1949.73, 1950.2,
        1950.82, 1952.04, 1953.39, 1950.06, 1949.29, 1949.5, 1950.17, 1949.32,
        1949.38, 1948.44, 1946.86, 1947.06, 1948.8, 1945.8, 1950.49, 1958.18,
        1963.0, 1960.64, 1961.91, 1963.07, 1958.44, 1957.32, 1958.41, 1959.11,
        1960.53, 1959.11, 1958.26, 1959.49, 1957.78, 1957.3, 1957.02, 1955.39,
        1954.99, 1953.95, 1954.1, 1948.26, 1947.39, 1946.13, 1942.63, 1942.07,
        1939.53, 1936.78, 1935.96, 1935.43, 1936.12, 1938.62
    ]
    
    print("=" * 60)
    print("Gold Futures Price Prediction (Version 4)")
    print("=" * 60)
    
    # Basic prediction
    prediction, confidence = predict_next_gold_price(test_data)
    print(f"\nLast Price: ${test_data[-1]:.2f}")
    print(f"Predicted Next Price: ${prediction:.2f}")
    print(f"Confidence Level: {confidence}%")
    
    # Detailed analysis
    print("\n" + "-" * 60)
    print("Detailed Analysis:")
    print("-" * 60)
    analysis = predict_with_analysis(test_data)
    for key, value in analysis.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Test constraint: verify prediction respects downward trend
    print("\n" + "-" * 60)
    print("Constraint Verification:")
    print("-" * 60)
    last_change = test_data[-1] - test_data[-2]
    if last_change < 0:
        if prediction <= test_data[-1]:
            print("✓ PASS: Prediction respects downward trend constraint")
            print(f"  (Prediction {prediction} <= Last Price {test_data[-1]})")
        else:
            print("✗ FAIL: Prediction violates downward trend constraint")
    else:
        print("  Last movement was upward - no constraint applied")
    
    print("\n" + "=" * 60)
