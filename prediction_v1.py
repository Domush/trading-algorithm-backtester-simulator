def predict(prices, past):
    """
    Predicts the next gold price using a feedback-adjusted linear regression.
    
    Args:
        prices (list): Historical actual prices (up to 100).
        past (list): The function's own previous 20 predictions.
        
    Returns:
        tuple: (adjusted_prediction, confidence_level)
    """
    if len(prices) < 5:
        return round(prices[-1], 2) if prices else 0.0, 50

    # 1. Calculate Error Bias (Self-Correction)
    # We look at the last few predictions to see if we are consistently high or low.
    bias_adjustment = 0
    if past:
        # We align the last prediction with the current actual price
        # past[-1] was the guess for prices[-1]
        errors = []
        depth = min(len(past), 5) # Look at last 5 errors
        for i in range(1, depth + 1):
            error = prices[-i] - past[-i]
            errors.append(error)
        
        # Average bias (if positive, we are underestimating; if negative, overestimating)
        bias_adjustment = sum(errors) / len(errors)

    # 2. Linear Regression for Trend (Next 5 points)
    n = 5
    x = list(range(n))
    y = prices[-n:]
    
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xx = sum(i*i for i in x)
    sum_xy = sum(x[i] * y[i] for i in range(n))
    
    # Calculate the Slope (the 'average of the trend')
    denominator = (n * sum_xx - sum_x**2)
    slope = (n * sum_xy - sum_x * sum_y) / denominator if denominator != 0 else 0
    
    # 3. Volatility Check (Distinct Trend Logic)
    mean_y = sum_y / n
    variance = sum((p - mean_y) ** 2 for p in y) / n
    std_dev = variance ** 0.5
    
    # A 'distinct trend' is where the projected 5-step movement 
    # exceeds the current market noise (std_dev).
    projected_movement = abs(slope * 5)
    is_distinct = projected_movement > (std_dev * 0.75)

    # 4. Final Prediction Calculation
    if is_distinct:
        # Prediction = Current Price + Trend Slope + Self-Correction
        raw_prediction = prices[-1] + slope + (bias_adjustment * 0.5)
    else:
        # Prediction = Current Price + Self-Correction (Mean Reversion)
        raw_prediction = prices[-1] + (bias_adjustment * 0.5)

    # 5. Confidence Level
    # Higher confidence if the trend is strong and our bias_adjustment is small (we are accurate)
    if std_dev == 0:
        confidence = 100
    else:
        # Ratio of trend strength to error/noise
        accuracy_factor = 1 / (1 + abs(bias_adjustment))
        trend_factor = projected_movement / std_dev
        conf_calc = (trend_factor * accuracy_factor) * 40
        confidence = max(1, min(100, int(conf_calc + 50)))

    return round(raw_prediction, 2), confidence
    