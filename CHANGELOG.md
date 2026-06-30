# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-06-30

### Added

- **Feedback-Adjusted Prediction Engine**: `prediction_v1.py` rewritten with a self-correcting linear regression algorithm. The prediction function now receives a rolling history of its own last 20 predictions, computes a bias adjustment from recent errors, and applies it to improve accuracy over time.
- **Clear All History Button**: New button in the History panel resets code history to the original v1 function, with a confirmation dialog before clearing.
- **Trades & Failed Predictions Tracking**: Backtests now count the number of trades entered and the number of failed predictions (predicted up but price went down), emitted via expanded progress and finished signals.
- **Timespan Reporting**: On backtest completion, the elapsed time covered by all ticks is calculated and reported (e.g., "2d 4h 30m") in the finished signal.
- **Configurable Tick Size / Tick Value**: `BacktestWorker` now accepts `tick_size` and `tick_value` as constructor parameters, replacing the previous hard-coded values.
- **Prediction & Confidence Offset Spinboxes**: Numeric spin boxes (`QDoubleSpinBox`) added alongside the existing sliders for precise offset entry, with `set_pred_offset()` and `set_conf_offset()` helpers to keep them in sync.
- **Max Ticks Settings Persistence**: The max ticks slider value is now saved to and restored from application settings.

### Changed

- **Prediction Function Signature**: The `predict` function contract changed from `predict(ohlcv_data)` (numpy OHLCV array) to `predict(close_prices, recent_predicted_prices)`, where `close_prices` is a plain Python list of up to 100 recent Close values and `recent_predicted_prices` is the last 20 values returned by the function itself.
- **Prediction Modules Consolidated**: `prediction_v2_1_10.py`, `prediction_v3.py`, `prediction_v4.py`, and `prediction_v5.py` removed. All algorithmic logic is now consolidated into the rewritten `prediction_v1.py`.
- **Max Ticks Slider Range**: Range changed from 10–30 (default 30) to 1–20 (default 5), keeping default backtests shorter and faster.
- **History Deduplication**: Code revisions are now deduplicated against the entire history (previously only against the last entry), preventing identical functions from appearing multiple times.
- **Project Renamed**: Application window title updated to "Trading Algorithm Backtester Pro".
- **Progress Signal Expanded**: Now emits `(timeframe, timestamp, actual, predicted, confidence_pct, is_success, invested, account_value, total_count, trades_count, failed_predictions)`.
- **Finished Signal Expanded**: Now emits `(timeframe, success_rate, total_ticks, timespan, trades_count, failed_predictions)`.

### Removed

- `prediction_v2_1_10.py`: Intermediate prediction model.
- `prediction_v3.py`: Local linear regression with dynamic shift adjustment.
- `prediction_v4.py`: Multi-method ensemble prediction model.
- `prediction_v5.py`: Conservative downward-trend-constrained prediction model.

## [1.1.0] - 2026-02-12

### Added

- **Advanced Prediction Modules**: Introduced `prediction_v3.py`, `prediction_v4.py`, and `prediction_v5.py` with sophisticated algorithms:
  - `prediction_v3.py`: Local linear regression-based prediction with dynamic shift adjustment.
  - `prediction_v4.py`: Multi-method ensemble prediction combining trend analysis, weighted moving averages, exponential smoothing, and momentum-based forecasting with downward trend constraints.
  - `prediction_v5.py`: Enhanced prediction model that guarantees conservative predictions during downward trends, combining linear regression, EMA, and Holt's method.
- **Confidence Scoring System**: Predictions now return a tuple `(predicted_price, confidence_level)` where confidence is calculated based on volatility, trend consistency, and data availability.
- **Futures Account Management**: Complete simulation of futures trading account with:
  - Configurable starting capital (default: $10,000).
  - Adjustable leverage (1x to 50x, default: 12x).
  - Maximum position value limits with optional forced liquidation.
  - Real-time account value tracking throughout backtests.
  - Dynamic contract allocation based on confidence levels (90%+ = 100%, 80-90% = 75%, 70-80% = 50%, 60-70% = 25%).
- **Prediction and Confidence Offset Sliders**: Fine-tune predictions and confidence levels with adjustable offsets:
  - Prediction offset: -10.0 to +10.0
  - Confidence offset: -80 to +80
- **Extended Historical Data Window**: Increased from 10 to 100 historical data points for more accurate predictions.
- **Account Auto-Abort**: Backtests automatically stop when account value falls below the cost of one contract.
- **Investment Tracking**: Real-time display of invested capital and account performance during backtests.
- **get_last_30_closes.py**: Utility script to fetch the last 30 closing prices for specified timeframes.

### Changed

- **BacktestWorker Enhanced**: Updated to handle confidence levels, account simulation, position sizing, and dynamic investment strategies.
- **Progress Signal Expanded**: Now emits `(timeframe, timestamp, actual, predicted, confidence, is_success, invested, account_value)`.
- **Window Size**: Default window size increased to 1900x1400 for better visibility of new features.
- **Dynamic Delay Algorithm**: Improved to handle longer backtests more efficiently (e.g., 11 days of 15-minute bars).
- **Prediction Function Contract**: User-defined functions must now return a tuple `(predicted_price, confidence_percentage)` instead of just a predicted price.

## [1.0.1] - 2026-02-12

### Changed

- Improved UI responsiveness by disabling the Stop button when no backtest is running (specifically when the 'Start Backtest' button is enabled).
- Enhanced button state management across code validation, history selection, and backtest execution.

## [1.0.0] - 2026-02-12

### Added

- Initial release of Gold Futures Backtester Pro.
- Support for multiple timeframes (1m to 1d).
- Real-time plotting with pyqtgraph.
- Integrated Python code editor with syntax highlighting.
- Code history tracking.
- Data resampling engine.
