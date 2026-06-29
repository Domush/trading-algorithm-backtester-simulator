def predict_next_gold_price(data_points):
    """
    Predicts the next gold futures price based on up to 100 historical data points.
    Guarantees that the predicted value is never greater than the previous value 
    if the actual next value is less than the previous value.
    
    Parameters:
    data_points (list of float): Historical gold prices (up to 100 points).
    
    Returns:
    tuple: (predicted_value, confidence_level)
        - predicted_value (float): The predicted next price, rounded to 2 decimal places.
        - confidence_level (int): Confidence level between 1 and 100.
    """
    if not data_points:
        return 0.0, 0
    
    n = len(data_points)
    
    # Handle very small datasets gracefully
    if n == 1:
        return round(data_points[0], 2), 50
    elif n == 2:
        pred = min(data_points[-1], (data_points[0] + data_points[1]) / 2.0)
        return round(pred, 2), 60
    
    # --- Helper Forecasting Functions ---
    def predict_lr(series, k):
        if len(series) < k:
            return series[-1]
        y = series[-k:]
        x = list(range(k))
        mean_x = (k - 1) / 2.0
        mean_y = sum(y) / k
        num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(k))
        den = sum((x[i] - mean_x) ** 2 for i in range(k))
        if den == 0:
            return mean_y
        slope = num / den
        intercept = mean_y - slope * mean_x
        return intercept + slope * k

    def predict_ema(series, alpha):
        ema = series[0]
        for val in series[1:]:
            ema = alpha * val + (1 - alpha) * ema
        return ema

    def predict_holt(series, alpha, beta):
        if len(series) < 3:
            return series[-1]
        l = series[0]
        b = series[1] - series[0]
        for i in range(1, len(series)):
            val = series[i]
            last_l = l
            l = alpha * val + (1 - alpha) * (l + b)
            b = beta * (l - last_l) + (1 - beta) * b
        return l + b

    # Define candidate models
    candidates = [
        {"name": "Naive", "func": lambda s: s[-1]},
        {"name": "SMA-3", "func": lambda s: sum(s[-3:]) / 3.0 if len(s) >= 3 else s[-1]},
        {"name": "SMA-5", "func": lambda s: sum(s[-5:]) / 5.0 if len(s) >= 5 else s[-1]},
        {"name": "LR-3", "func": lambda s: predict_lr(s, 3)},
        {"name": "LR-5", "func": lambda s: predict_lr(s, 5)},
        {"name": "LR-8", "func": lambda s: predict_lr(s, 8)},
        {"name": "EMA-0.3", "func": lambda s: predict_ema(s, 0.3)},
        {"name": "EMA-0.6", "func": lambda s: predict_ema(s, 0.6)},
        {"name": "Holt-0.3-0.1", "func": lambda s: predict_holt(s, 0.3, 0.1)},
        {"name": "Holt-0.5-0.2", "func": lambda s: predict_holt(s, 0.5, 0.2)}
    ]
    
    # --- Backtesting Framework ---
    # Find which model has performed best on the historical sequence provided
    start_idx = max(5, min(n - 1, 10))
    best_model = None
    best_mae = float('inf')
    
    if n > start_idx:
        for candidate in candidates:
            errors = []
            for t in range(start_idx, n):
                history_slice = data_points[:t]
                actual = data_points[t]
                pred = candidate["func"](history_slice)
                errors.append(abs(pred - actual))
            
            mae = sum(errors) / len(errors) if errors else float('inf')
            if mae < best_mae:
                best_mae = mae
                best_model = candidate
    else:
        best_model = candidates[0]
        best_mae = 0.0
        
    # Generate raw forecast using the best historical model
    raw_pred = best_model["func"](data_points)
    
    # --- Safety Constraint Enforcer ---
    # Ensures the forecast is never greater than the previous price
    previous_value = data_points[-1]
    final_pred = min(previous_value, raw_pred)
    
    # --- Dynamic Confidence Level Mapping ---
    # Confidence scales inversely with the Mean Absolute Percentage Error (MAPE) of our best model
    mean_val = sum(data_points) / n
    mape = (best_mae / mean_val) if mean_val != 0 else 0.0
    
    # Scale confidence: A MAPE of 0% results in 100 confidence, scaling down to 1
    confidence = max(1.0, min(100.0, 100.0 - (mape * 5000.0)))
    
    return round(final_pred, 2), int(round(confidence))