I want to create a backtesting app in python which allows me to backtest predictive python functions on historical gold futures prices. I want you to create a step-by-step plan I can pass to an LLM to create the app with every feature specified below:

# App features
- A Modern, elegant GUI with dark theme
 - A graph which shows historical gold prices and overlays the predicted prices returned by the predictive function
 - A code box which allows me to edit the python function's prediction code, with code highlighting
  - Every backtest should save a history of changes to the prediction code. Only add a history entry if the code changed. Selecting a past prediction entry should update the prediction code back to the entry.
  - The backtest button should remain disabled until the 'update' button is clicked, where the python code is checked for errors. If no errors, then enable the backtest button.
 - A 'backtest' button which executes the backtest
- Allows me to select 1min, 5min, 15min, 30min, 1hour, and/or 1day price datasets for backtesting. Each will be displayed in their own graph.
- The dataset is a very large (376MB) .csv file. The same 1min interval csv file will be used for all backtesting timeframes. (eg: 5min will use every 5th entry)
 - Optional: allow importing of a csv file and converting to a faster/optimized way of storing the data
- The graphs should be updated in realtime, as the backtesting progresses
- Each timescale graph should show the predictive success rate as a percentage
- I should be able to select a lower and upper 'success' threshold (eg: within 2 if actual next value is lower, within 10 if actual next value is higher)
- The prediction function should get passed in a list of the last 10 open, close, high, low, and volume values from the dataset. Only the code within the prediction function should be editable, the function itself should be static. numpy should be available to the function.
- The backtests should auto abort on a per-timespan basis if any predicted value deviates outside a user-configurable range (eg: more than 100 from actual)
- The app should utilize multithreading

# Implementation Plan: Gold Futures Backtester

## Phase 1: Environment & Data Engine
1.  **Libraries:** Use `PySide6` for GUI, `pyqtgraph` for charting, `pandas` for data handling, `pyarrow` for fast storage (Feather/Parquet), and `pygments` for the code highlighter.
2.  **Data Optimization:** Create a data-loading module that:
    *   Imports the 376MB CSV.
    *   Converts it to **Feather** format (`df.to_feather()`) for near-instant loading on subsequent runs.
    *   Includes a resampling function that takes the 1-minute base data and generates 5m, 15m, 30m, 1h, and 1d OHLCV (Open, High, Low, Close, Volume) dataframes.

## Phase 2: Modern Dark GUI Design
1.  **Theme:** Use a dark stylesheet (e.g., `qt-material` or a custom CSS-like QSS) with a slate/charcoal background and accent colors for price lines.
2.  **Layout:**
    *   **Sidebar:** Contains dataset selection (multi-select), threshold inputs (upper/lower success, abort range), and a "Backtest History" list.
    *   **Center:** A tabbed interface where each tab contains a `pyqtgraph.PlotWidget` for a specific selected timeframe.
    *   **Bottom/Right:** A code editor box for the predictive function.
3.  **Code Box:** Implement a `QPlainTextEdit` with a custom `QSyntaxHighlighter` using `pygments`.
    *   Add an **"Update"** button that runs `compile(code, '<string>', 'exec')` to check for syntax errors.
    *   Enable the **"Backtest"** button only after a successful "Update."

## Phase 3: The Backtesting Engine (Multithreading)
1.  **Worker Thread:** Create a `QThread` (or `QRunnable`) for the backtest to prevent the GUI from freezing.
2.  **Prediction Logic:**
    *   The engine iterates through the dataframe.
    *   For each step, it slices the last 10 rows and passes them to the user's function as a NumPy array.
    *   **Success Rate Logic:** Compare the function's return value against the next actual price using the user-defined lower/upper thresholds.
    *   **Abort Logic:** If `abs(predicted - actual) > user_threshold`, stop the specific timeframe backtest.
3.  **Real-time Updates:** Use Qt **Signals** to emit the latest (index, actual_price, predicted_price, success_bool) back to the GUI. The GUI should update the `pyqtgraph` plots and the "Success Rate %" label dynamically.

## Phase 4: Persistence & History
1.  **Code History:** Use a simple list or SQLite database to store `(timestamp, code_string)`.
    *   Before starting a backtest, check if the current code matches the last entry. If not, append it.
    *   When a user clicks an item in the "Backtest History" list, populate the code box with that version.

---

# LLM Prompt for Code Generation

**Copy and paste the following into your LLM:**

> "Act as a Senior Python Developer. Create a desktop application using PySide6 and pyqtgraph for backtesting gold futures.
>
> **Core Requirements:**
> 1. **GUI:** Modern dark theme. Use `pyqtgraph` for the main charts because of the large dataset.
> 2. **Data:** The app must load a 1min Gold Futures CSV (approx 376MB). Include a 'Convert' feature to save this as a Feather file for speed. Implement resampling to 5m, 15m, 30m, 1h, and 1d.
> 3. **Code Editor:** A syntax-highlighted editor box for a function: `def predict(ohlcv_data):`. Use `pygments` for highlighting.
>    - Add an 'Update' button to validate code via `compile()`.
>    - 'Backtest' button is disabled until 'Update' succeeds.
>    - Maintain a history of code changes; clicking a history entry reverts the editor code.
> 4. **Multithreading:** Use `QThread` and Signals to run backtests in the background. Update the graphs and 'Success %' labels in realtime as the loop progresses.
> 5. **Logic:**
>    - Pass the last 10 rows of OHLCV data to the user function.
>    - Allow user to set 'Success Thresholds' (different for up/down moves).
>    - Implement an 'Auto-abort' if a prediction deviates from the actual price by a user-defined amount.
> 6. **Visualization:** Each selected timeframe should have its own graph tab, showing the actual price line and a scatter/line overlay for predictions.
>
> Please provide the full modular code including the GUI layout, the Threaded Worker, and the Custom Highlighter."

---

### Pro Tips for Success:
*   **Memory Management:** With a 376MB file, loading the whole CSV into RAM is fine (it will take ~1-2GB in a Pandas DataFrame), but ensure you use `float32` instead of `float64` for the OHLCV columns to save space.
*   **Graphing Speed:** For the "Realtime Update," don't redraw the whole graph every step. Use `plotDataItem.setData()` to append only the new points for maximum performance.
*   **Code Execution:** Use `exec()` carefully. Since this is a local app for your own use, it is safe, but always wrap the user's code execution in a `try...except` block inside the worker thread to prevent a single bad prediction from crashing the whole app.