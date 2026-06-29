import sys
import os

# Explicitly set QT_API for pyqtgraph
os.environ['QT_API'] = 'pyside6'

import time
import traceback
import math
import pandas as pd
import numpy as np
import PySide6 # Import PySide6 before pyqtgraph
import pyqtgraph as pg
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QRadioButton, QButtonGroup,
                             QStackedWidget, QPushButton, QLabel,
                             QPlainTextEdit, QLineEdit, QFormLayout, QGroupBox,
                             QListWidget, QSplitter, QMessageBox, QCheckBox,
                             QMenu, QSlider, QFileDialog, QSpinBox)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QSettings, QTimer
from PySide6.QtGui import QFont, QColor

from data_engine import DataEngine
from highlighter import PygmentsHighlighter

# --- UI Components ---
class RangeSlider(QWidget):
    valueChanged = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.start_slider = QSlider(Qt.Horizontal)
        self.end_slider = QSlider(Qt.Horizontal)

        # Style sliders to be more compact
        slider_style = """
            QSlider::handle:horizontal {
                background: #3498db;
                border: 1px solid #555;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border_radius: 7px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #333;
                height: 4px;
                background: #121212;
                margin: 2px 0;
            }
        """
        self.start_slider.setStyleSheet(slider_style)
        self.end_slider.setStyleSheet(slider_style)

        layout.addWidget(self.start_slider)
        layout.addWidget(self.end_slider)

        self.start_slider.valueChanged.connect(self._handle_start_change)
        self.end_slider.valueChanged.connect(self._handle_end_change)

    def _handle_start_change(self, value):
        if value > self.end_slider.value():
            self.end_slider.setValue(value)
        self.valueChanged.emit(self.start_slider.value(), self.end_slider.value())

    def _handle_end_change(self, value):
        if value < self.start_slider.value():
            self.start_slider.setValue(value)
        self.valueChanged.emit(self.start_slider.value(), self.end_slider.value())

    def setRange(self, min_val, max_val):
        self.start_slider.blockSignals(True)
        self.end_slider.blockSignals(True)
        self.start_slider.setRange(min_val, max_val)
        self.end_slider.setRange(min_val, max_val)
        self.start_slider.setValue(min_val)
        self.end_slider.setValue(max_val)
        self.start_slider.blockSignals(False)
        self.end_slider.blockSignals(False)
        self.valueChanged.emit(min_val, max_val)

    def values(self):
        return self.start_slider.value(), self.end_slider.value()

    def setValues(self, low, high):
        self.start_slider.setValue(low)
        self.end_slider.setValue(high)

# --- Worker Thread ---
class BacktestWorker(QThread):
    # Signals: timeframe, timestamp, actual, predicted, confidence_pct, is_success, invested, account_value
    progress = Signal(str, float, float, float, float, bool, float, float)
    finished = Signal(str, float)
    error = Signal(str)
    aborted = Signal(str, str)
    abort_point = Signal(str, float, float)

    def __init__(self, data, predict_code, timeframe, thresholds, toggles, abort_range, start_capital, leverage, max_position_value, sell_above_max, prediction_offset, confidence_offset):
        super().__init__()
        self.data = data
        self.predict_code = predict_code
        self.timeframe = timeframe
        self.thresholds = thresholds
        self.toggles = toggles # {'up': bool, 'down': bool}
        self.abort_range = abort_range
        self.start_capital = start_capital
        self.leverage = leverage
        self.max_position_value = max_position_value
        self.sell_above_max = sell_above_max
        self.prediction_offset = prediction_offset
        self.confidence_offset = confidence_offset
        self.tick_value = 10.0
        self.tick_size = 0.1
        self.account_value = float(self.start_capital)
        self.position = 0
        self.entry_contracts = 0
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            # Prepare execution environment
            local_env = {'np': np, 'numpy': np}
            exec(self.predict_code, local_env)

            # The function should be named 'predict' as per requirement
            if 'predict' not in local_env:
                # Try to find any function if 'predict' is missing
                funcs = [v for k, v in local_env.items() if callable(v)]
                if not funcs:
                    self.error.emit("No function found in code.")
                    return
                predict_func = funcs[0]
            else:
                predict_func = local_env['predict']

            success_count = 0
            total_count = 0

            # Start from row 100
            if len(self.data) > 100:
                start_idx = 99
                start_ts = float(self.data.index[start_idx].timestamp())
                start_actual = self.data.iloc[start_idx]['Close']
                self.progress.emit(self.timeframe, start_ts, start_actual, start_actual, 0.0, False, 0.0, self.account_value)

            for i in range(100, len(self.data)):
                if not self._is_running:
                    break

                window = self.data.iloc[i-100:i]
                actual = self.data.iloc[i]['Close']
                ts = float(self.data.index[i].timestamp())
                prev_close = self.data.iloc[i-1]['Close']
                ohlcv_data = window[['Open', 'High', 'Low', 'Close', 'Volume']].values

                try:
                    prediction_result = predict_func(ohlcv_data)
                except Exception as e:
                    self.error.emit(f"Runtime error in prediction: {str(e)}")
                    return

                if isinstance(prediction_result, tuple) and len(prediction_result) == 2:
                    predicted, confidence_pct = prediction_result
                else:
                    self.error.emit("Predict function must return (prediction, confidence_pct).")
                    return

                predicted = float(predicted) + self.prediction_offset
                confidence_pct = float(confidence_pct) + self.confidence_offset
                confidence_pct = max(0.0, min(100.0, confidence_pct))

                # Check for abort
                if abs(predicted - actual) > self.abort_range:
                    self.aborted.emit(self.timeframe, f"Abort: Dev {abs(predicted-actual):.2f} > {self.abort_range}")
                    return

                # Success logic
                is_success = False
                diff = abs(predicted - actual)
                if actual >= prev_close: # Up/Flat
                    if self.toggles['up']:
                        if diff <= self.thresholds['up']:
                            is_success = True
                    else:
                        is_success = True # Ignore threshold if disabled
                else: # Down
                    if self.toggles['down']:
                        if diff <= self.thresholds['down']:
                            is_success = True
                    else:
                        is_success = True # Ignore threshold if disabled

                if is_success:
                    success_count += 1
                total_count += 1

                # Account simulation: fixed tick futures contract, long-only.
                contract_cost = 0.85 * prev_close
                if self.position == 0:
                    if self.account_value < contract_cost:
                        self.abort_point.emit(self.timeframe, ts, self.account_value)
                        self.aborted.emit(self.timeframe, f"Stopped: account value below one contract cost (${contract_cost:.2f})")
                        return

                    base_contracts = int(self.account_value // contract_cost)
                    if confidence_pct >= 90.0:
                        confidence_factor = 1.0
                    elif confidence_pct >= 80.0:
                        confidence_factor = 0.75
                    elif confidence_pct >= 70.0:
                        confidence_factor = 0.50
                    elif confidence_pct >= 60.0:
                        confidence_factor = 0.25
                    else:
                        confidence_factor = 0.0

                    num_contracts = int(math.ceil(base_contracts * confidence_factor)) if confidence_factor > 0 else 0
                    if self.max_position_value > 0:
                        max_contracts = int(self.max_position_value // contract_cost)
                        num_contracts = min(num_contracts, max_contracts)

                    if predicted > prev_close and num_contracts > 0:
                        self.position = 1
                        self.entry_contracts = num_contracts
                        invested_amount = self.entry_contracts * contract_cost
                    else:
                        invested_amount = 0.0
                else:
                    num_contracts = self.entry_contracts
                    if self.max_position_value > 0 and self.sell_above_max:
                        max_contracts = int(self.max_position_value // contract_cost)
                        if num_contracts > max_contracts:
                            num_contracts = max_contracts
                            if num_contracts == 0:
                                self.position = 0
                                self.entry_contracts = 0
                                invested_amount = 0.0
                            else:
                                self.entry_contracts = num_contracts
                                invested_amount = num_contracts * contract_cost
                        else:
                            invested_amount = num_contracts * contract_cost if num_contracts > 0 else 0.0
                    else:
                        invested_amount = num_contracts * contract_cost if num_contracts > 0 else 0.0

                    if predicted <= prev_close:
                        self.position = 0
                        self.entry_contracts = 0
                        invested_amount = 0.0

                ticks = (actual - prev_close) / self.tick_size
                pnl = self.position * ticks * self.tick_value * num_contracts * self.leverage
                self.account_value += pnl

                self.progress.emit(self.timeframe, ts, actual, predicted, confidence_pct, is_success, invested_amount, self.account_value)

                if self.position == 1 and self.account_value < contract_cost:
                    self.abort_point.emit(self.timeframe, ts, self.account_value)
                    self.aborted.emit(self.timeframe, f"Stopped: account value below one contract cost (${contract_cost:.2f})")
                    return

                # Dynamic delay based on total_count to keep the app responsive as data grows.
                # Increase the ramp-up significantly for longer interval runs (e.g. 11 days of 15m bars).
                if i % 10 == 0:
                    dynamic_pause = 0.02 + min(0.8, (total_count / 25.0) * 0.015)
                    time.sleep(dynamic_pause)
                else:
                    time.sleep(0.001)

            final_rate = (success_count / total_count * 100) if total_count > 0 else 0
            self.finished.emit(self.timeframe, final_rate)

        except Exception as e:
            self.error.emit(f"Worker Exception: {str(e)}")

# --- Main Window ---
class GoldBacktester(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gold Futures Backtester Pro")
        self.resize(1900, 1400)

        self.settings = QSettings("GoldPredictive", "Backtester")
        self.data_engine = DataEngine("data/XAU_1m_data.csv")
        self.workers = {}
        self.code_history = [] # List of tuples: (timestamp, code)
        self.timeframes = ['1m', '5m', '15m', '30m', '1h', '1d']
        self.selected_timeframe = '1h'
        self.plot_components = {}
        self.account_plot_components = {}

        # Plot data storage
        self.plot_data = {tf: {
            'indices': [],
            'actuals': [],
            'predicts_x': [],
            'predicts_y': [],
            'confidences': [],
            'colors': [], # List of brush colors
            'successes': [], # List of bools
            'success_count': 0
        } for tf in self.timeframes}

        self.account_data = {tf: {
            'timestamps': [],
            'invested': [],
            'account_values': [],
            'buy_timestamps': [],
            'buy_values': [],
            'sell_timestamps': [],
            'sell_values': [],
            'abort_timestamps': [],
            'abort_values': [],
            'last_account_value': None,
            'last_invested': 0.0
        } for tf in self.timeframes}

        self.account_plot_items = {}
        self.invested_plot_items = {}
        self.buy_plot_items = {}
        self.sell_plot_items = {}
        self.abort_plot_items = {}

        self.init_ui()
        self.apply_dark_theme()

        # Load saved settings and history
        self.load_settings()

        # If no code in settings/history, load from file
        if not self.code_editor.toPlainText().strip():
            self.load_initial_code()

        # Attempt to load data on startup
        QTimer.singleShot(100, self.on_load_data)

    def closeEvent(self, event):
        self.save_settings()
        super().closeEvent(event)

    def save_settings(self):
        # Save thresholds and other UI states
        self.settings.setValue("up_thresh", self.up_thresh.text())
        self.settings.setValue("down_thresh", self.down_thresh.text())
        self.settings.setValue("up_toggle", self.up_toggle.isChecked())
        self.settings.setValue("down_toggle", self.down_toggle.isChecked())
        self.settings.setValue("abort_thresh", self.abort_thresh.text())
        self.settings.setValue("prediction_offset", self.pred_offset_slider.value())
        self.settings.setValue("confidence_offset", self.conf_offset_slider.value())
        self.settings.setValue("start_capital", self.start_capital.text())
        self.settings.setValue("leverage", self.leverage_input.value())
        self.settings.setValue("max_position_value", self.max_position_value.text())
        self.settings.setValue("sell_above_max", self.sell_above_max_checkbox.isChecked())
        self.settings.setValue("active_timeframe", self.selected_timeframe)
        self.settings.setValue("window_width", self.width())
        self.settings.setValue("window_height", self.height())
        self.settings.setValue("current_code", self.code_editor.toPlainText())

        # Save history
        self.settings.setValue("history", self.code_history)

        # Save date range if data is loaded
        if hasattr(self, 'start_date_ref'):
            low, high = self.range_slider.values()
            start_date = self.start_date_ref + pd.Timedelta(days=low)
            end_date = self.start_date_ref + pd.Timedelta(days=high)
            self.settings.setValue("start_date", start_date.isoformat())
            self.settings.setValue("end_date", end_date.isoformat())

    def load_settings(self):
        self.up_thresh.setText(self.settings.value("up_thresh", "10.0"))
        self.down_thresh.setText(self.settings.value("down_thresh", "2.0"))

        # Load toggles
        up_toggle = self.settings.value("up_toggle", "true")
        self.up_toggle.setChecked(str(up_toggle).lower() == 'true')
        down_toggle = self.settings.value("down_toggle", "true")
        self.down_toggle.setChecked(str(down_toggle).lower() == 'true')

        self.abort_thresh.setText(self.settings.value("abort_thresh", "100.0"))
        offset_value = int(self.settings.value("prediction_offset", 0))
        self.pred_offset_slider.setValue(offset_value)
        self.pred_offset_value.setText(f"{offset_value / 10:.1f}")
        conf_offset_value = int(self.settings.value("confidence_offset", 0))
        self.conf_offset_slider.setValue(conf_offset_value)
        self.conf_offset_value.setText(str(conf_offset_value))
        self.start_capital.setText(self.settings.value("start_capital", "10000"))
        leverage = int(self.settings.value("leverage", 12))
        self.leverage_input.setValue(leverage)
        self.max_position_value.setText(self.settings.value("max_position_value", "0"))
        sell_above_max = self.settings.value("sell_above_max", "false")
        self.sell_above_max_checkbox.setChecked(str(sell_above_max).lower() == 'true')

        active_timeframe = self.settings.value("active_timeframe", self.selected_timeframe)
        if active_timeframe in self.tf_radios:
            self.selected_timeframe = active_timeframe
            self.tf_radios[active_timeframe].setChecked(True)
            self.plot_stack.setCurrentIndex(self.timeframes.index(self.selected_timeframe))

        width = int(self.settings.value("window_width", 1900))
        height = int(self.settings.value("window_height", 1400))
        self.resize(width, height)

        # Restore current code
        current_code = self.settings.value("current_code", "")
        if current_code:
            self.code_editor.setPlainText(current_code)

        # Restore history
        saved_history = self.settings.value("history", [])
        if saved_history:
            self.code_history = saved_history
            for timestamp, code in self.code_history:
                self.history_list.addItem(f"Revision {self.history_list.count()+1} - {timestamp}")

        # Load date range
        self.saved_start_date = self.settings.value("start_date", None)
        self.saved_end_date = self.settings.value("end_date", None)

    def update_offset_label(self, value):
        self.pred_offset_value.setText(f"{value / 10:.1f}")

    def update_conf_offset_label(self, value):
        self.conf_offset_value.setText(str(value))

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Splitter for Sidebar and Main Area
        splitter = QSplitter(Qt.Horizontal)

        # --- Sidebar ---
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar.setMaximumWidth(350)

        # Data Loading Group
        data_group = QGroupBox("Data Management")
        data_vbox = QVBoxLayout()
        self.load_btn = QPushButton("Load CSV")
        self.load_btn.clicked.connect(self.on_load_data)
        self.convert_btn = QPushButton("Convert to Feather")
        self.convert_btn.clicked.connect(self.on_convert_feather)
        self.convert_btn.hide()

        # Status Indicator
        self.status_label = QLabel("Data: Not Loaded")
        self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")

        data_vbox.addWidget(self.load_btn)
        data_vbox.addWidget(self.convert_btn)
        data_vbox.addWidget(self.status_label)
        data_group.setLayout(data_vbox)
        sidebar_layout.addWidget(data_group)

        # Backtest Range Group
        range_group = QGroupBox("Backtest Date Range")
        range_vbox = QVBoxLayout()

        self.range_slider = RangeSlider()
        self.range_label = QLabel("All Data")
        self.range_label.setStyleSheet("font-size: 9pt; color: #3498db;")
        self.range_label.setWordWrap(True)

        range_vbox.addWidget(self.range_label)
        range_vbox.addWidget(self.range_slider)
        range_group.setLayout(range_vbox)
        sidebar_layout.addWidget(range_group)

        self.range_slider.valueChanged.connect(self.on_range_changed)

        # Check feather existence and hide convert button if it exists
        if os.path.exists(self.data_engine.feather_path):
            self.convert_btn.hide()

        # Timeframe Selection
        tf_group = QGroupBox("Select Timeframe")
        tf_vbox = QVBoxLayout()
        self.tf_radios = {}
        self.tf_button_group = QButtonGroup(self)
        self.tf_button_group.setExclusive(True)
        for tf in self.timeframes:
            rb = QRadioButton(tf)
            self.tf_radios[tf] = rb
            self.tf_button_group.addButton(rb)
            tf_vbox.addWidget(rb)
            if tf == self.selected_timeframe:
                rb.setChecked(True)
        tf_group.setLayout(tf_vbox)
        sidebar_layout.addWidget(tf_group)
        self.tf_button_group.buttonClicked.connect(self.on_timeframe_changed)

        # Thresholds
        thresh_group = QGroupBox("Success Thresholds")
        thresh_layout = QVBoxLayout()

        # Up Thresh Row
        up_hbox = QHBoxLayout()
        self.up_toggle = QCheckBox("Up Move (<=):")
        self.up_toggle.setChecked(True)
        self.up_toggle.setToolTip("Enable upper success threshold for upward predictions.")
        self.up_thresh = QLineEdit("10.0")
        self.up_thresh.setToolTip("Maximum allowed error for upward predictions to count as success.")
        up_hbox.addWidget(self.up_toggle)
        up_hbox.addWidget(self.up_thresh)

        # Down Thresh Row
        down_hbox = QHBoxLayout()
        self.down_toggle = QCheckBox("Down Move (<=):")
        self.down_toggle.setChecked(True)
        self.down_toggle.setToolTip("Enable lower success threshold for downward predictions.")
        self.down_thresh = QLineEdit("2.0")
        self.down_thresh.setToolTip("Maximum allowed error for downward predictions to count as success.")
        down_hbox.addWidget(self.down_toggle)
        down_hbox.addWidget(self.down_thresh)

        # Abort Thresh Row
        abort_hbox = QHBoxLayout()
        abort_label = QLabel("Auto-Abort (>):")
        abort_label.setToolTip("Label for the auto-abort threshold.")
        self.abort_thresh = QLineEdit("100.0")
        self.abort_thresh.setToolTip("Stop backtests when prediction error exceeds this value.")
        abort_hbox.addWidget(abort_label)
        abort_hbox.addWidget(self.abort_thresh)

        # Prediction Offset Slider
        offset_hbox = QHBoxLayout()
        offset_label = QLabel("Pred Offset:")
        offset_label.setToolTip("Offset the predicted value by a fixed amount.")
        self.pred_offset_slider = QSlider(Qt.Horizontal)
        self.pred_offset_slider.setRange(-100, 100)
        self.pred_offset_slider.setValue(0)
        self.pred_offset_slider.setTickInterval(10)
        self.pred_offset_slider.setTickPosition(QSlider.TicksBelow)
        self.pred_offset_slider.setSingleStep(1)
        self.pred_offset_value = QLabel("0.0")
        self.pred_offset_slider.valueChanged.connect(self.update_offset_label)
        offset_hbox.addWidget(offset_label)
        offset_hbox.addWidget(self.pred_offset_slider)
        offset_hbox.addWidget(self.pred_offset_value)

        conf_offset_hbox = QHBoxLayout()
        conf_offset_label = QLabel("Conf Offset:")
        conf_offset_label.setToolTip("Offset the confidence percentage by a fixed amount.")
        self.conf_offset_slider = QSlider(Qt.Horizontal)
        self.conf_offset_slider.setRange(-80, 80)
        self.conf_offset_slider.setValue(0)
        self.conf_offset_slider.setTickInterval(5)
        self.conf_offset_slider.setTickPosition(QSlider.TicksBelow)
        self.conf_offset_slider.setSingleStep(1)
        self.conf_offset_slider.setPageStep(5)
        self.conf_offset_slider.setTracking(True)
        self.conf_offset_value = QLabel("0")
        self.conf_offset_slider.valueChanged.connect(self.update_conf_offset_label)
        conf_offset_hbox.addWidget(conf_offset_label)
        conf_offset_hbox.addWidget(self.conf_offset_slider)
        conf_offset_hbox.addWidget(self.conf_offset_value)

        thresh_layout.addLayout(up_hbox)
        thresh_layout.addLayout(down_hbox)
        thresh_layout.addLayout(abort_hbox)
        thresh_layout.addLayout(offset_hbox)
        thresh_layout.addLayout(conf_offset_hbox)
        thresh_group.setLayout(thresh_layout)
        sidebar_layout.addWidget(thresh_group)

        # Futures Account
        account_group = QGroupBox("Futures Account")
        account_layout = QVBoxLayout()

        start_hbox = QHBoxLayout()
        self.start_capital = QLineEdit("10000")
        self.start_capital.setPlaceholderText("Starting Capital")
        start_label = QLabel("Start Capital:")
        start_hbox.addWidget(start_label)
        start_hbox.addWidget(self.start_capital)

        leverage_hbox = QHBoxLayout()
        self.leverage_input = QSpinBox()
        self.leverage_input.setRange(1, 50)
        self.leverage_input.setValue(12)
        self.leverage_input.setSuffix("x")
        leverage_label = QLabel("Leverage:")
        leverage_hbox.addWidget(leverage_label)
        leverage_hbox.addWidget(self.leverage_input)

        maxpos_hbox = QHBoxLayout()
        self.max_position_value = QLineEdit("0")
        self.max_position_value.setPlaceholderText("Max Position Value")
        self.max_position_value.setToolTip("Maximum notional position size before forced reduction.")
        maxpos_label = QLabel("Max Position:")
        maxpos_hbox.addWidget(maxpos_label)
        maxpos_hbox.addWidget(self.max_position_value)

        self.sell_above_max_checkbox = QCheckBox("Sell positions above max value")
        self.sell_above_max_checkbox.setChecked(False)
        self.sell_above_max_checkbox.setToolTip("If checked, positions above max value will be sold down to the limit.")

        account_layout.addLayout(start_hbox)
        account_layout.addLayout(leverage_hbox)
        account_layout.addLayout(maxpos_hbox)
        account_layout.addWidget(self.sell_above_max_checkbox)
        account_group.setLayout(account_layout)
        sidebar_layout.addWidget(account_group)

        # History
        hist_group = QGroupBox("Code History")
        hist_vbox = QVBoxLayout()
        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self.on_history_select)
        self.history_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.history_list.customContextMenuRequested.connect(self.on_history_context_menu)
        hist_vbox.addWidget(self.history_list)

        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout()

        count_hbox = QHBoxLayout()
        self.export_count_slider = QSlider(Qt.Horizontal)
        self.export_count_slider.setRange(10, 100)
        self.export_count_slider.setValue(50)
        self.export_count_slider.setTickInterval(10)
        self.export_count_slider.setTickPosition(QSlider.TicksBelow)
        self.export_count_slider.valueChanged.connect(self.update_export_count_label)
        self.export_count_label = QLabel("Export count: 50")
        self.export_count_label.setStyleSheet("color: #dcdcdc;")
        count_hbox.addWidget(self.export_count_label)
        export_layout.addLayout(count_hbox)
        export_layout.addWidget(self.export_count_slider)

        button_hbox = QHBoxLayout()
        self.copy_export_btn = QPushButton("Copy to Clipboard")
        self.copy_export_btn.clicked.connect(self.copy_export_to_clipboard)
        self.csv_export_btn = QPushButton("Export to CSV File")
        self.csv_export_btn.clicked.connect(self.export_history_to_csv)
        button_hbox.addWidget(self.copy_export_btn)
        button_hbox.addWidget(self.csv_export_btn)
        export_layout.addLayout(button_hbox)

        export_group.setLayout(export_layout)
        hist_vbox.addWidget(export_group)

        hist_group.setLayout(hist_vbox)
        sidebar_layout.addWidget(hist_group)

        sidebar_layout.addStretch()

        # --- Main Area ---
        main_area = QWidget()
        main_area_layout = QVBoxLayout(main_area)

        # Chart stack for single selected timeframe
        self.plot_stack = QStackedWidget()
        self.plot_widgets = {}
        self.plot_items = {} # actual price line
        self.predict_items = {} # prediction scatter/line
        self.rate_labels = {}

        for tf in self.timeframes:
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)

            # Create Plot with Date Axis
            axis = pg.DateAxisItem(orientation='bottom')
            pw = pg.PlotWidget(title=f"Gold Futures - {tf}", axisItems={'bottom': axis})
            pw.setBackground('#121212')
            pw.showGrid(x=True, y=True, alpha=0.3)
            pw.addLegend()

            # Enable tooltips on the plot
            pw.setMouseEnabled(x=True, y=True)

            # Success Rate Label
            rate_label = QLabel("Success Rate: 0.00%")
            rate_label.setTextFormat(Qt.RichText)
            rate_label.setStyleSheet("font-size: 14pt; color: #dcdcdc;")
            rate_label.setAlignment(Qt.AlignRight)

            tab_layout.addWidget(rate_label)
            tab_layout.addWidget(pw)

            self.plot_widgets[tf] = pw
            self.rate_labels[tf] = rate_label
            self.plot_stack.addWidget(tab)

            # Lines
            self.plot_items[tf] = pw.plot(pen=pg.mkPen(color='#3498db', width=1.5), name="Actual")
            self.predict_items[tf] = pw.plot(pen=None, symbol='o', symbolSize=6, symbolBrush='#e74c3c', name="Predicted")

            account_axis = pg.DateAxisItem(orientation='bottom')
            account_pw = pg.PlotWidget(title=f"Account Metrics - {tf}", axisItems={'bottom': account_axis})
            account_pw.setBackground('#121212')
            account_pw.showGrid(x=True, y=True, alpha=0.3)
            account_pw.addLegend()
            # Keep the account chart x-axis aligned with the main price chart.
            account_pw.setXLink(pw)
            self.account_plot_items[tf] = account_pw.plot(pen=pg.mkPen(color='#f1c40f', width=2, style=Qt.DashLine), name="Account Value")
            self.invested_plot_items[tf] = account_pw.plot(pen=pg.mkPen(color='#8e44ad', width=1, style=Qt.DotLine), name="Invested")
            self.buy_plot_items[tf] = account_pw.plot(pen=None, symbol='o', symbolSize=10, symbolBrush='#2ecc71', name='Buy')
            self.sell_plot_items[tf] = account_pw.plot(pen=None, symbol='o', symbolSize=10, symbolBrush='#e74c3c', name='Sell')
            self.abort_plot_items[tf] = account_pw.plot(pen=None, symbol='x', symbolSize=14, symbolBrush='#ff0000', name='Abort')
            tab_layout.addWidget(account_pw)

            # Create crosshair lines for main price plot
            vLine = pg.InfiniteLine(angle=90, movable=False, pen='#666')
            hLine = pg.InfiniteLine(angle=0, movable=False, pen='#666')
            pw.addItem(vLine, ignoreBounds=True)
            pw.addItem(hLine, ignoreBounds=True)

            # Text item for main plot tooltip
            label = pg.TextItem(anchor=(0, 1), color='#fff', fill='#333', border='#555')
            label.setZValue(100)
            pw.addItem(label, ignoreBounds=True)

            # Store components for main plot hover logic
            self.plot_components[tf] = {
                'vLine': vLine,
                'hLine': hLine,
                'label': label,
                'vb': pw.getViewBox()
            }

            # Create crosshair lines for account chart
            account_vLine = pg.InfiniteLine(angle=90, movable=False, pen='#666')
            account_hLine = pg.InfiniteLine(angle=0, movable=False, pen='#666')
            account_pw.addItem(account_vLine, ignoreBounds=True)
            account_pw.addItem(account_hLine, ignoreBounds=True)

            account_label = pg.TextItem(anchor=(0, 1), color='#fff', fill='#333', border='#555')
            account_label.setZValue(100)
            account_pw.addItem(account_label, ignoreBounds=True)

            self.account_plot_components[tf] = {
                'vLine': account_vLine,
                'hLine': account_hLine,
                'label': account_label,
                'vb': account_pw.getViewBox()
            }

            # Connect hover signals using lambdas to pass the timeframe
            pw.scene().sigMouseMoved.connect(lambda pos, t=tf: self.on_mouse_moved(pos, t))
            account_pw.scene().sigMouseMoved.connect(lambda pos, t=tf: self.on_account_mouse_moved(pos, t))

        self.plot_stack.setCurrentIndex(self.timeframes.index(self.selected_timeframe))
        main_area_layout.addWidget(self.plot_stack, stretch=2)

        # Code Editor Section
        editor_group = QGroupBox("Predictive Function Editor")
        editor_layout = QVBoxLayout()

        self.code_editor = QPlainTextEdit()
        self.code_editor.setFont(QFont("Consolas", 11))
        self.highlighter = PygmentsHighlighter(self.code_editor.document())

        btn_layout = QHBoxLayout()
        self.update_btn = QPushButton("Update / Validate")
        self.update_btn.clicked.connect(self.on_update_code)
        self.backtest_btn = QPushButton("Start Backtest")
        self.backtest_btn.setEnabled(False)
        self.backtest_btn.clicked.connect(self.on_start_backtest)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.on_stop_backtest)

        btn_layout.addWidget(self.update_btn)
        btn_layout.addWidget(self.backtest_btn)
        btn_layout.addWidget(self.stop_btn)

        editor_layout.addWidget(self.code_editor)
        editor_layout.addLayout(btn_layout)
        editor_group.setLayout(editor_layout)

        main_area_layout.addWidget(editor_group, stretch=1)

        splitter.addWidget(sidebar)
        splitter.addWidget(main_area)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

    def apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #333;
                margin-top: 1.1em;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
            QPushButton {
                background-color: #333;
                border: 1px solid #555;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #444;
            }
            QPushButton:disabled {
                color: #666;
                background-color: #222;
            }
            QPlainTextEdit {
                background-color: #121212;
                color: #dcdcdc;
                border: 1px solid #333;
            }
            QLineEdit {
                background-color: #121212;
                border: 1px solid #333;
                padding: 4px;
            }
            QRadioButton {
                color: #e0e0e0;
                spacing: 8px;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #888;
                border-radius: 8px;
                background: #121212;
            }
            QRadioButton::indicator:checked {
                background: #3498db;
                border: 1px solid #3498db;
            }
            QListWidget {
                background-color: #121212;
                border: 1px solid #333;
            }
        """)

    def load_initial_code(self):
        try:
            with open("prediction_v1.py", "r") as f:
                content = f.read()

            # The requirement is def predict(ohlcv_data).
            # The original code uses predict_next_number(data) and expects a 1D list of prices.
            # We transform it to be compatible:
            old_sig = "def predict_next_number(data):"
            new_sig = "def predict(ohlcv_data):\n    # Extract Close prices (index 3) from OHLCV array\n    data = ohlcv_data[:, 3].tolist()"

            content = content.replace(old_sig, new_sig)

            # Remove any example usage at the bottom to keep the editor clean
            if "# Example usage" in content:
                content = content.split("# Example usage")[0].strip()

            self.code_editor.setPlainText(content)
        except Exception:
            self.code_editor.setPlainText("def predict(ohlcv_data):\n    # ohlcv_data is 10x5 numpy array: [Open, High, Low, Close, Volume]\n    return ohlcv_data[-1, 3] + 1.0")

    @Slot()
    def on_load_data(self):
        # Determine the file path
        file_path = None

        # If the button was clicked, always show the dialog
        if self.sender() == self.load_btn:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open CSV Data", "data", "CSV Files (*.csv)"
            )
            if not file_path:
                return
        else:
            # Startup or internal call - check if default exists
            default_csv = "data/XAU_1m_data.csv"
            default_feather = default_csv.replace('.csv', '.feather')
            if os.path.exists(default_feather) or os.path.exists(default_csv):
                file_path = default_csv
            else:
                return # No default data found, wait for user action

        try:
            self.data_engine.set_csv_path(file_path)
            self.data_engine.load_data()
            self.status_label.setText("Data: Loaded")
            self.status_label.setStyleSheet("color: #27ae60; font-weight: bold;")

            # Update date range slider
            if self.data_engine.df is not None and not self.data_engine.df.empty:
                min_date = self.data_engine.df.index.min().normalize()
                max_date = self.data_engine.df.index.max().normalize()

                self.start_date_ref = min_date
                num_days = (max_date - min_date).days

                self.range_slider.setRange(0, num_days)
                self.update_range_label(0, num_days)

                # Apply saved range if available
                if self.saved_start_date and self.saved_end_date:
                    try:
                        s_dt = pd.to_datetime(self.saved_start_date)
                        e_dt = pd.to_datetime(self.saved_end_date)

                        # Calculate day offsets from min_date
                        low = (s_dt.normalize() - min_date).days
                        high = (e_dt.normalize() - min_date).days

                        # Clamp to range
                        low = max(0, min(num_days, low))
                        high = max(0, min(num_days, high))

                        self.range_slider.setValues(low, high)
                        self.update_range_label(low, high)
                    except Exception:
                        pass # Ignore if saved dates are invalid

            # If we just loaded a CSV (and feather doesn't exist yet), show convert
            if self.data_engine.feather_path and not os.path.exists(self.data_engine.feather_path):
                self.convert_btn.show()
            else:
                self.convert_btn.hide()
        except Exception as e:
            self.status_label.setText("Data: Error")
            self.status_label.setStyleSheet("color: #c0392b; font-weight: bold;")
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")

    @Slot(int, int)
    def on_range_changed(self, low, high):
        self.update_range_label(low, high)

    def update_range_label(self, low, high):
         if not hasattr(self, 'start_date_ref'):
             return
         start_date = self.start_date_ref + pd.Timedelta(days=low)
         end_date = self.start_date_ref + pd.Timedelta(days=high)
         self.range_label.setText(f"Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    @Slot()
    def on_convert_feather(self):
        try:
            self.data_engine.convert_to_feather()
            self.convert_btn.hide()
            self.status_label.setText("Data: Feather Ready")
            self.status_label.setStyleSheet("color: #27ae60; font-weight: bold;")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Conversion failed: {str(e)}")

    @Slot()
    def on_update_code(self):
        code = self.code_editor.toPlainText()
        try:
            compile(code, '<string>', 'exec')
            self.backtest_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.update_btn.setStyleSheet("background-color: #27ae60;")

            # Save to history if changed
            # Extract only the code from the tuples in self.code_history
            if not self.code_history or self.code_history[-1][1] != code:
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                self.code_history.append((timestamp, code))
                self.history_list.addItem(f"Revision {len(self.code_history)} - {timestamp}")
                self.save_settings() # Persist history change
        except Exception as e:
            self.backtest_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.update_btn.setStyleSheet("background-color: #c0392b;")
            QMessageBox.warning(self, "Syntax Error", str(e))

    def on_history_select(self, item):
        idx = self.history_list.row(item)
        if 0 <= idx < len(self.code_history):
            self.code_editor.setPlainText(self.code_history[idx][1])
            self.backtest_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.update_btn.setStyleSheet("")

    def on_history_context_menu(self, pos):
        item = self.history_list.itemAt(pos)
        if item:
            menu = QMenu()
            delete_action = menu.addAction("Delete Entry")
            action = menu.exec(self.history_list.mapToGlobal(pos))
            if action == delete_action:
                idx = self.history_list.row(item)
                self.code_history.pop(idx)
                self.history_list.takeItem(idx)

                # Update revision numbers in display
                for i in range(self.history_list.count()):
                    list_item = self.history_list.item(i)
                    timestamp = self.code_history[i][0]
                    list_item.setText(f"Revision {i+1} - {timestamp}")

                self.save_settings()

    def get_export_dataframe(self):
        if self.data_engine.df is None:
            self.on_load_data()
            if self.data_engine.df is None:
                return pd.DataFrame()

        selected_tf = self.selected_timeframe
        low_days, high_days = self.range_slider.values()
        if hasattr(self, 'start_date_ref'):
            start_dt = self.start_date_ref + pd.Timedelta(days=low_days)
            end_dt = self.start_date_ref + pd.Timedelta(days=high_days)
            start_dt = start_dt.replace(hour=0, minute=0, second=0)
            end_dt = end_dt.replace(hour=23, minute=59, second=59)
        else:
            start_dt = self.data_engine.df.index.min()
            end_dt = self.data_engine.df.index.max()

        df_tf = self.data_engine.get_resampled_data(selected_tf)
        df_range = df_tf[(df_tf.index >= start_dt) & (df_tf.index <= end_dt)]
        count = self.export_count_slider.value()
        if df_range.empty:
            return df_range
        return df_range.tail(count)

    def update_export_count_label(self, value):
        self.export_count_label.setText(f"Export count: {value}")

    def copy_export_to_clipboard(self):
        df_export = self.get_export_dataframe()
        if df_export.empty:
            QMessageBox.warning(self, "Export Error", "No data available for export in the selected range.")
            return
        close_prices = df_export['Close'].astype(str).tolist()
        clipboard_text = ",".join(close_prices)
        QApplication.clipboard().setText(clipboard_text)
        QMessageBox.information(self, "Export Complete", f"Copied {len(df_export)} close prices to clipboard.")

    def export_history_to_csv(self):
        df_export = self.get_export_dataframe()
        if df_export.empty:
            QMessageBox.warning(self, "Export Error", "No data available for export in the selected range.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "export.csv", "CSV Files (*.csv)")
        if not file_path:
            return
        try:
            df_export.to_csv(file_path)
            QMessageBox.information(self, "Export Complete", f"Exported {len(df_export)} rows to {file_path}.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to write CSV: {str(e)}")

    @Slot()
    def on_start_backtest(self):
        if not self.data_engine.df is not None:
            self.on_load_data()

        selected_tf = self.selected_timeframe
        if selected_tf not in self.timeframes:
            QMessageBox.warning(self, "Warning", "Select a valid timeframe.")
            return

        # Reset plots
        self.plot_data[selected_tf] = {
            'indices': [],
            'actuals': [],
            'predicts_x': [],
            'predicts_y': [],
            'confidences': [],
            'colors': [],
            'successes': [],
            'success_count': 0
        }
        self.plot_items[selected_tf].setData([], [])
        self.predict_items[selected_tf].setData([], [])
        self.rate_labels[selected_tf].setStyleSheet("font-size: 14pt; color: #dcdcdc;")
        self.rate_labels[selected_tf].setText("Success Rate: 0.00%")

        # Autoscale charts at the beginning of the backtest
        try:
            self.plot_items[selected_tf].getViewBox().enableAutoRange()
            self.account_plot_items[selected_tf].getViewBox().enableAutoRange()
        except Exception:
            pass

        code = self.code_editor.toPlainText()
        thresholds = {
            'up': float(self.up_thresh.text()),
            'down': float(self.down_thresh.text())
        }
        toggles = {
            'up': self.up_toggle.isChecked(),
            'down': self.down_toggle.isChecked()
        }
        abort_range = float(self.abort_thresh.text())

        # Get selected range
        low_days, high_days = self.range_slider.values()
        if hasattr(self, 'start_date_ref'):
            start_dt = self.start_date_ref + pd.Timedelta(days=low_days)
            end_dt = self.start_date_ref + pd.Timedelta(days=high_days)
            # Ensure start_dt is start of day and end_dt is end of day
            start_dt = start_dt.replace(hour=0, minute=0, second=0)
            end_dt = end_dt.replace(hour=23, minute=59, second=59)
        else:
            # Fallback to full range if data not loaded properly
            start_dt = self.data_engine.df.index.min()
            end_dt = self.data_engine.df.index.max()

        start_capital = float(self.start_capital.text())
        self.current_start_capital = start_capital
        leverage = float(self.leverage_input.value())
        max_position_value = float(self.max_position_value.text()) if self.max_position_value.text().strip() else 0.0
        sell_above_max = self.sell_above_max_checkbox.isChecked()
        prediction_offset = self.pred_offset_slider.value() / 10.0
        confidence_offset = self.conf_offset_slider.value()

        self.account_data[selected_tf] = {
            'timestamps': [],
            'invested': [],
            'account_values': [],
            'buy_timestamps': [],
            'buy_values': [],
            'sell_timestamps': [],
            'sell_values': [],
            'abort_timestamps': [],
            'abort_values': [],
            'final_timestamp': None,
            'final_account_value': None,
            'final_invested': None,
            'last_account_value': None,
            'last_invested': 0.0
        }
        self.account_plot_items[selected_tf].setData([], [])
        self.invested_plot_items[selected_tf].setData([], [])
        self.buy_plot_items[selected_tf].setData([], [])
        self.sell_plot_items[selected_tf].setData([], [])
        self.abort_plot_items[selected_tf].setData([], [])

        df_tf = self.data_engine.get_resampled_data(selected_tf)

        # Filter by date range
        df_tf = df_tf[(df_tf.index >= start_dt) & (df_tf.index <= end_dt)]

        if len(df_tf) < 11:
            self.rate_labels[selected_tf].setText("Insufficient Data")
        else:
            worker = BacktestWorker(df_tf, code, selected_tf, thresholds, toggles, abort_range, start_capital, leverage, max_position_value, sell_above_max, prediction_offset, confidence_offset)
            worker.progress.connect(self.update_plot)
            worker.finished.connect(self.on_worker_finished)
            worker.error.connect(self.on_worker_error)
            worker.aborted.connect(self.on_worker_aborted)
            worker.abort_point.connect(self.on_abort_point)

            self.workers[selected_tf] = worker
            worker.start()

        self.backtest_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    @Slot()
    def on_timeframe_changed(self, button):
        selected_tf = button.text()
        if selected_tf in self.timeframes:
            self.selected_timeframe = selected_tf
            self.plot_stack.setCurrentIndex(self.timeframes.index(selected_tf))

    @Slot(str, float, float, float, float, bool, float, float)
    def update_plot(self, tf, ts, actual, predicted, confidence_pct, is_success, invested_amount, account_value):
        d = self.plot_data[tf]
        previous_ts = d['indices'][-1] if d['indices'] else ts
        d['indices'].append(ts)
        d['actuals'].append(actual)
        d['predicts_x'].append(ts)
        d['predicts_y'].append(predicted)
        d['confidences'].append(confidence_pct)
        d['successes'].append(is_success)

        acc = self.account_data[tf]
        prev_invested = acc['last_invested'] if acc['last_invested'] is not None else 0.0
        plot_account_value = acc['last_account_value'] if acc['last_account_value'] is not None else account_value
        plot_invested = acc['last_invested'] if acc['last_invested'] is not None else invested_amount

        shift_seconds = self._get_account_shift_seconds(tf)
        shifted_ts = ts + shift_seconds

        acc['timestamps'].append(shifted_ts)
        acc['invested'].append(plot_invested)
        acc['account_values'].append(plot_account_value)

        if invested_amount > 0 and prev_invested == 0.0:
            acc['buy_timestamps'].append(shifted_ts)
            acc['buy_values'].append(plot_account_value)
        elif invested_amount == 0.0 and prev_invested > 0.0:
            acc['sell_timestamps'].append(shifted_ts)
            acc['sell_values'].append(plot_account_value)

        acc['last_account_value'] = account_value
        acc['last_invested'] = invested_amount
        acc['final_timestamp'] = ts
        acc['final_account_value'] = account_value
        acc['final_invested'] = invested_amount

        # Color code: Green for success, Red for failure
        color = '#27ae60' if is_success else '#e74c3c'
        d['colors'].append(pg.mkBrush(color))

        if is_success:
            d['success_count'] += 1

        self.plot_items[tf].setData(d['indices'], d['actuals'])
        # For larger backtests, avoid rendering a scatter symbol for every predicted point.
        if len(d['indices']) > 5000:
            self.predict_items[tf].setData(
                x=d['predicts_x'],
                y=d['predicts_y'],
                pen=pg.mkPen(color='#e74c3c', width=1),
                symbol=None
            )
        else:
            self.predict_items[tf].setData(
                x=d['predicts_x'],
                y=d['predicts_y'],
                symbol='o',
                symbolBrush=list(d['colors'])
            )

        final_ts = acc['final_timestamp']
        account_timestamps = list(acc['timestamps'])
        account_values = list(acc['account_values'])
        invested_values = list(acc['invested'])
        if final_ts is not None:
            account_timestamps.append(final_ts)
            account_values.append(acc['final_account_value'])
            invested_values.append(acc['final_invested'])

        self.account_plot_items[tf].setData(account_timestamps, account_values)
        self.invested_plot_items[tf].setData(account_timestamps, invested_values)
        self.buy_plot_items[tf].setData(acc['buy_timestamps'], acc['buy_values'])
        self.sell_plot_items[tf].setData(acc['sell_timestamps'], acc['sell_values'])
        self.abort_plot_items[tf].setData(acc['abort_timestamps'], acc['abort_values'])

        rate = (d['success_count'] / len(d['indices']) * 100)
        self.rate_labels[tf].setStyleSheet("font-size: 14pt; color: #dcdcdc;")
        account_color = '#27ae60' if account_value > getattr(self, 'current_start_capital', 0.0) else '#ff4d4f' if account_value < getattr(self, 'current_start_capital', 0.0) else '#dcdcdc'
        self.rate_labels[tf].setText(
            f"Success Rate: <span style='color: #27ae60;'>{rate:.2f}%</span> | "
            f"Account: <span style='color: {account_color};'>${account_value:,.2f}</span>"
        )

    @Slot(str, float)
    def on_worker_finished(self, tf, rate):
        final_value = 0.0
        if tf in self.account_data and self.account_data[tf]['account_values']:
            final_value = self.account_data[tf]['account_values'][-1]
        self.rate_labels[tf].setStyleSheet("font-size: 14pt; color: #dcdcdc;")
        account_color = '#27ae60' if final_value > getattr(self, 'current_start_capital', 0.0) else '#ff4d4f' if final_value < getattr(self, 'current_start_capital', 0.0) else '#dcdcdc'
        self.rate_labels[tf].setText(
            f"FINAL Success Rate: <span style='color: #27ae60;'>{rate:.2f}%</span> | "
            f"Final Account: <span style='color: {account_color};'>${final_value:,.2f}</span>"
        )
        if tf in self.workers:
            del self.workers[tf]
        if not self.workers:
            self.backtest_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

    @Slot(str)
    def on_worker_error(self, msg):
        QMessageBox.critical(self, "Worker Error", msg)
        self.on_stop_backtest()

    @Slot(str, str)
    def on_worker_aborted(self, tf, reason):
        self.rate_labels[tf].setText(f"ABORTED: {reason}")
        self.rate_labels[tf].setStyleSheet("font-size: 14pt; color: #ff0000;")
        if tf in self.workers:
            del self.workers[tf]
        if not self.workers:
            self.backtest_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

    @Slot(str, float, float)
    def on_abort_point(self, tf, ts, account_value):
        acc = self.account_data.get(tf)
        if acc is None:
            return
        # Abort should show at the actual final tick, not on the shifted account series.
        acc['abort_timestamps'].append(ts)
        acc['abort_values'].append(account_value)
        self.abort_plot_items[tf].setData(acc['abort_timestamps'], acc['abort_values'])

    @Slot()
    def on_stop_backtest(self):
        for tf, worker in self.workers.items():
            worker.stop()
            worker.wait()
        self.workers.clear()
        self.backtest_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _get_closest_index(self, indices, timestamp):
        import bisect
        idx_pos = bisect.bisect_left(indices, timestamp)
        if idx_pos == 0:
            return 0
        if idx_pos == len(indices):
            return len(indices) - 1

        left_dist = abs(timestamp - indices[idx_pos - 1])
        right_dist = abs(indices[idx_pos] - timestamp)
        return idx_pos - 1 if left_dist < right_dist else idx_pos

    def _get_account_shift_seconds(self, tf):
        # Shift account chart data one full timeframe interval to the left.
        if tf == '1m':
            return -60
        if tf == '5m':
            return -300
        if tf == '15m':
            return -900
        if tf == '30m':
            return -1800
        if tf == '1h':
            return -3600
        if tf == '1d':
            return -86400
        return -60

    def _hide_tooltips(self, tf):
        for components in (self.plot_components.get(tf), self.account_plot_components.get(tf)):
            if components:
                components['vLine'].hide()
                components['hLine'].hide()
                components['label'].hide()

    def _sync_tooltips(self, tf, timestamp, source='main'):
        if tf not in self.plot_data or not self.plot_data[tf]['indices']:
            self._hide_tooltips(tf)
            return
        if tf not in self.account_data or not self.account_data[tf]['timestamps']:
            self._hide_tooltips(tf)
            return

        if source == 'account':
            account_timestamps = list(self.account_data[tf]['timestamps'])
            account_values = list(self.account_data[tf]['account_values'])
            invested_values = list(self.account_data[tf]['invested'])
            final_ts = self.account_data[tf].get('final_timestamp')
            if final_ts is not None:
                account_timestamps.append(final_ts)
                account_values.append(self.account_data[tf].get('final_account_value', account_values[-1] if account_values else 0.0))
                invested_values.append(self.account_data[tf].get('final_invested', invested_values[-1] if invested_values else 0.0))

            closest_account_idx = self._get_closest_index(account_timestamps, timestamp)
            if closest_account_idx < 0 or closest_account_idx >= len(account_timestamps):
                self._hide_tooltips(tf)
                return

            indices = self.plot_data[tf]['indices']
            price_idx = max(0, closest_account_idx - 1)
            actual_ts = account_timestamps[closest_account_idx]
            account_val = account_values[closest_account_idx]
            invested_val = invested_values[closest_account_idx]
            prev_account_val = account_values[closest_account_idx - 1] if closest_account_idx > 0 else account_val
            delta = account_val - prev_account_val
        else:
            account_timestamps = list(self.account_data[tf]['timestamps'])
            account_values = list(self.account_data[tf]['account_values'])
            invested_values = list(self.account_data[tf]['invested'])
            final_ts = self.account_data[tf].get('final_timestamp')
            if final_ts is not None:
                account_timestamps.append(final_ts)
                account_values.append(self.account_data[tf].get('final_account_value', account_values[-1] if account_values else 0.0))
                invested_values.append(self.account_data[tf].get('final_invested', invested_values[-1] if invested_values else 0.0))

            indices = self.plot_data[tf]['indices']
            closest_account_idx = self._get_closest_index(account_timestamps, timestamp)
            closest_idx = self._get_closest_index(indices, timestamp)
            if closest_idx < 0 or closest_idx >= len(indices):
                self._hide_tooltips(tf)
                return

            actual_ts = indices[closest_idx]
            account_val = account_values[closest_account_idx] if closest_account_idx < len(account_values) else account_values[-1]
            invested_val = invested_values[closest_account_idx] if closest_account_idx < len(invested_values) else invested_values[-1]
            prev_account_val = account_values[closest_account_idx - 1] if closest_account_idx > 0 else account_val
            delta = account_val - prev_account_val

            price_idx = closest_idx

        actuals = self.plot_data[tf]['actuals']
        predicts_y = self.plot_data[tf]['predicts_y']
        confidences = self.plot_data[tf]['confidences']
        successes = self.plot_data[tf]['successes']

        actual_val = actuals[price_idx]
        pred_val = predicts_y[price_idx]
        confidence_pct = confidences[price_idx]
        is_success = successes[price_idx]
        prev_actual = actuals[price_idx - 1] if price_idx > 0 else actual_val
        pred_change = pred_val - prev_actual
        actual_change = actual_val - prev_actual
        diff = pred_val - actual_val

        def get_color(val):
            return "#27ae60" if val >= 0 else "#e74c3c"

        time_str = pd.to_datetime(actual_ts, unit='s').strftime('%Y-%m-%d %H:%M')

        main_text = f"<span style='color: white'>{time_str}</span><br>"
        main_text += f"<span style='color: white'>P: {pred_val:.2f} </span>"
        main_text += f"<span style='color: {get_color(pred_change)}'>({pred_change:+.2f})</span><br>"
        main_text += f"<span style='color: white'>A: {actual_val:.2f} </span>"
        main_text += f"<span style='color: {get_color(actual_change)}'>({actual_change:+.2f})</span><br>"
        main_text += f"<span style='color: white'>Conf: {confidence_pct:.1f}%</span><br>"
        main_text += f"<span style='color: white'>≅: </span>"
        main_text += f"<span style='color: {get_color(diff)}'>{diff:+.2f} </span>"
        main_text += ("<span style='color: #27ae60'>✔</span>" if is_success else "<span style='color: #e74c3c'>✘</span>")

        account_text = f"<span style='color: white'>{time_str}</span><br>"
        account_text += f"<span style='color: white'>Account: ${account_val:,.2f}</span><br>"
        account_text += f"<span style='color: {get_color(delta)}'>Δ: {delta:+.2f}</span><br>"
        account_text += f"<span style='color: white'>Invested: ${invested_val:,.2f}</span>"

        main_components = self.plot_components.get(tf)
        account_components = self.account_plot_components.get(tf)

        if main_components:
            main_components['label'].setHtml(main_text)
            main_components['vLine'].setPos(actual_ts)
            main_components['hLine'].setPos(actual_val)
            view_rect = main_components['vb'].viewRect()
            anchor_x = 1 if actual_ts > (view_rect.left() + view_rect.width() * 0.7) else 0
            main_components['label'].setAnchor((anchor_x, 0.5))
            main_components['label'].setPos(actual_ts, view_rect.center().y())
            main_components['vLine'].show()
            main_components['hLine'].show()
            main_components['label'].show()

        if account_components:
            account_components['label'].setHtml(account_text)
            account_components['vLine'].setPos(actual_ts)
            account_components['hLine'].setPos(account_val)
            view_rect = account_components['vb'].viewRect()
            anchor_x = 1 if actual_ts > (view_rect.left() + view_rect.width() * 0.7) else 0
            account_components['label'].setAnchor((anchor_x, 0.5))
            account_components['label'].setPos(actual_ts, view_rect.center().y())
            account_components['vLine'].show()
            account_components['hLine'].show()
            account_components['label'].show()

    def on_mouse_moved(self, pos, tf):
        if tf not in self.plot_data or not self.plot_data[tf]['indices']:
            self._hide_tooltips(tf)
            return

        components = self.plot_components.get(tf)
        if not components:
            self._hide_tooltips(tf)
            return

        vb = components['vb']
        if not vb.sceneBoundingRect().contains(pos):
            self._hide_tooltips(tf)
            return

        mousePoint = vb.mapSceneToView(pos)
        self._sync_tooltips(tf, mousePoint.x())

    def on_account_mouse_moved(self, pos, tf):
        if tf not in self.account_data or not self.account_data[tf]['timestamps']:
            self._hide_tooltips(tf)
            return

        components = self.account_plot_components.get(tf)
        if not components:
            self._hide_tooltips(tf)
            return

        vb = components['vb']
        if not vb.sceneBoundingRect().contains(pos):
            self._hide_tooltips(tf)
            return

        mousePoint = vb.mapSceneToView(pos)
        self._sync_tooltips(tf, mousePoint.x(), source='account')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GoldBacktester()
    window.show()
    sys.exit(app.exec())
