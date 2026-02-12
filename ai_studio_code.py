def predict_next_number(data_points):
    """
    Predicts the next number in a sequence using a dynamic sliding 
    window linear regression based on the most recent data.
    
    Args:
        data_points (list): A list of numerical values (accepts up to 10).
        
    Returns:
        float: The predicted next value in the sequence.
    """
    # Ensure we have at least 2 points to establish a trend
    if not data_points:
        return 0.0
    if len(data_points) < 2:
        return float(data_points[-1])

    # Use only the last 10 points as permitted by the requirements
    window = data_points[-10:]
    n = len(window)
    
    # Create the X-axis (time/index) for the window
    x_coords = list(range(n))
    y_coords = window
    
    # Calculate components for Linear Regression (y = mx + b)
    sum_x = sum(x_coords)
    sum_y = sum(y_coords)
    sum_xx = sum(x**2 for x in x_coords)
    sum_xy = sum(x * y for x, y in zip(x_coords, y_coords))
    
    # Calculate denominator for slope formula
    denominator = (n * sum_xx - sum_x**2)
    
    # If points are identical (vertical line/division by zero), return the last value
    if denominator == 0:
        return float(window[-1])
    
    # Calculate Slope (m) and Intercept (b)
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    
    # Predict the value for the next index (n)
    prediction = (slope * n) + intercept
    
    return round(prediction, 2)

# Example Usage:
# sequence = [4102.85, 4104.1, 4103.95, 4104.85, 4104.84, 4101.66, 4102.33, 4102.46, 4100.37, 4099.0, 4101.26, 4099.4]
# print(predict_next_number(sequence))