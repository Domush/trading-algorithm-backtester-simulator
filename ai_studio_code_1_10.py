import numpy as np

def predict_next_number(data):
    """
    Predicts the next number in the sequence within the specified error bounds:
    - If prediction < actual: error <= 1
    - If prediction > actual: error <= 10
    
    This is achieved by targeting a prediction approximately 4.5 units 
    above the expected value to maximize the hit rate within the 11-point window.
    """
    if not data:
        return 0.0
    
    # Use the last 10 data points as allowed by the requirements
    # If fewer than 10 are available, use what we have
    points = np.array(data[-10:])
    n = len(points)
    
    if n < 2:
        return round(points[-1] + 4.5, 2)

    # 1. Trend Analysis: Linear Regression on the recent window
    x = np.arange(n)
    y = points
    slope, intercept = np.polyfit(x, y, 1)
    
    # Predict the raw next value based on current trend
    raw_prediction = slope * n + intercept
    
    # 2. Seasonality/Lag Check: Every 7th data point
    # Financial data often has a weekly (7-period) cycle.
    # We check the 7-period difference to adjust for 'gap' movements.
    if len(data) >= 8:
        lag_7_diff = data[-1] - data[-8]
        # Dampen the lag effect to avoid overreacting to noise
        raw_prediction = 0.7 * raw_prediction + 0.3 * (data[-1] + (lag_7_diff / 7))

    # 3. Asymmetric Bias Adjustment
    # The requirement [Actual-1, Actual+10] implies we should target 4.5 
    # points above the most likely actual value.
    # This allows the actual value to rise by up to 5.5 (within 1 of pred)
    # or fall by up to 5.5 (within 10 of pred).
    final_prediction = raw_prediction + 4.5
    
    # 4. Volatility Guard
    # If the last point is significantly below the local mean (a 'dip'),
    # we adjust higher to catch the likely mean-reversion (recovery).
    local_mean = np.mean(points)
    if points[-1] < local_mean - np.std(points):
        final_prediction = max(final_prediction, local_mean + 2.0)

    return round(final_prediction, 2)

# Example usage with the last 10 points of your provided dataset:
# data_sample = [4103.95, 4104.85, 4104.84, 4101.66, 4102.33, 4102.46, 4100.37, 4099.0, 4101.26, 4099.4]
# print(predict_next_number(data_sample))