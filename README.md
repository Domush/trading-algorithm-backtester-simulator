# Trading Algorithm Backtester Pro

<img width="1961" height="1400" alt="image" src="https://github.com/user-attachments/assets/e82e827b-c11d-409a-99d3-0e5750a77bcd" />

A modern, elegant desktop application for backtesting predictive Python functions on historical prices with advanced account simulation and confidence-based trading strategies.

## Features

- **Modern GUI**: Elegant dark-themed interface built with PySide6.
- **Real-time Visualization**: Dynamic charts using `pyqtgraph` that update as the backtest progresses.
- **Multi-Timeframe Support**: Backtest across 1m, 5m, 15m, 30m, 1h, and 1d timeframes simultaneously.
- **Integrated Code Editor**: Syntax-highlighted editor with validation and history tracking.
- **Multithreaded Engine**: High-performance backtesting that keeps the UI responsive.
- **Data Management**: Efficient loading of large CSV datasets with Feather format conversion for speed.
- **Customizable Logic**: User-defined success thresholds and auto-abort parameters.
- **Futures Account Simulation**: Complete trading account simulation with:
  - Configurable starting capital and leverage (1x-50x)
  - Dynamic position sizing based on confidence levels
  - Maximum position value limits with optional forced liquidation
  - Real-time P&L and account value tracking
  - Auto-abort when account falls below contract cost
- **Confidence-Based Trading**: Predictions include confidence levels that drive position sizing:
  - 90%+ confidence: 100% of available capital
  - 80-90%: 75% of capital
  - 70-80%: 50% of capital
  - 60-70%: 25% of capital
  - Below 60%: No position
- **Prediction & Confidence Tuning**: Adjust prediction values and confidence levels with real-time sliders for strategy optimization.
- **Advanced Prediction Models**: Includes multiple prediction algorithms (v1-v5) with trend analysis, moving averages, exponential smoothing, and downward trend constraints.
- **Extended Historical Context**: Uses up to 100 historical data points for more accurate predictions.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/trading-algorithm-backtester-simulator.git
cd trading-algorithm-backtester-simulator
```

1. Install dependencies:

```bash
pip install pyside6 pyqtgraph pandas numpy pygments feather-format
```

1. Ensure a historical price CSV-formatted OHLCV data file exists in `data`.

## Usage

1. Run the application:

```bash
python main.py
```

1. Configure account parameters:
   - Set starting capital (default: $10,000)
   - Adjust leverage (1x-50x, default: 12x)
   - Optionally set maximum position value
2. Select the desired timeframes in the sidebar.
3. Write or edit your predictive function in the editor. The function must follow the signature:

   ```python
   def predict(ohlcv_data):
         # ohlcv_data is a numpy array of up to 100 historical rows (OHLCV format)
         # Must return a tuple: (predicted_price, confidence_percentage)
         # Example:
         predicted_price = 1950.50
         confidence = 75  # 0-100
         return predicted_price, confidence
   ```

4. Adjust thresholds and offsets as needed:
   - **Up/Down thresholds**: Define success criteria for predictions
   - **Auto-abort threshold**: Stop if prediction error exceeds this value
   - **Prediction offset**: Add/subtract a fixed amount to all predictions
   - **Confidence offset**: Adjust confidence levels up or down
5. Click **Update / Validate** to check for syntax errors.
6. Click **Start Backtest** to begin. Watch real-time updates of:
   - Price predictions vs actual values
   - Account value and invested capital
   - Success rates across timeframes
7. Use the **Stop** button to halt a running backtest.

## UI Controls

- **Start Backtest**: Enabled only after successful code validation.
- **Stop**: Enabled only while a backtest is in progress.
- **Update / Validate**: Compiles the current code and saves it to history if changed.
- **Backtest History**: Click any revision to revert the editor to that version.
- **Prediction Offset Slider**: Adjust predictions by -10.0 to +10.0 points.
- **Confidence Offset Slider**: Adjust confidence levels by -80 to +80 percentage points.
- **Sell Above Max Checkbox**: Automatically reduce positions that exceed maximum position value.

## Project Structure

- `main.py`: Main application entry point and UI logic with account simulation.
- `data_engine.py`: Data loading and resampling engine.
- `highlighter.py`: Syntax highlighter for the code editor.
- `prediction_v1.py`: Basic prediction model (baseline).
- `prediction_v2_1_10.py`: Intermediate prediction model.
- `prediction_v3.py`: Local linear regression-based prediction with dynamic shift adjustment.
- `prediction_v4.py`: Advanced multi-method ensemble with trend analysis, moving averages, and confidence scoring.
- `prediction_v5.py`: Conservative prediction model with downward trend constraints.
- `get_last_30_closes.py`: Utility script to fetch recent closing prices for any timeframe.
- `data/`: Directory for historical datasets (CSV and Feather formats).
- `PLAN.md`: Development roadmap and feature planning.
- `CHANGELOG.md`: Detailed version history.
