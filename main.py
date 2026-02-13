import sys
import os

# Explicitly set QT_API for pyqtgraph
os.environ['QT_API'] = 'pyside6'

import time
import traceback
import pandas as pd
import numpy as np
import PySide6 # Import PySide6 before pyqtgraph
import pyqtgraph as pg
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTabWidget, QPushButton, QLabel,
                             QPlainTextEdit, QLineEdit, QFormLayout, QGroupBox,
                             QListWidget, QSplitter, QMessageBox, QCheckBox,
                             QMenu, QSlider, QFileDialog)
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
    # Signals: timeframe, timestamp, actual, predicted, is_success
    progress = Signal(str, float, float, float, bool)
    finished = Signal(str, float)
    error = Signal(str)
    aborted = Signal(str, str)

    def __init__(self, data, predict_code, timeframe, thresholds, toggles, abort_range):
        super().__init__()
        self.data = data
        self.predict_code = predict_code
        self.timeframe = timeframe
        self.thresholds = thresholds
        self.toggles = toggles # {'up': bool, 'down': bool}
        self.abort_range = abort_range
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

            # Start from row 10
            for i in range(10, len(self.data)):
                if not self._is_running:
                    break

                window = self.data.iloc[i-10:i]
                actual = self.data.iloc[i]['Close']
                ts = float(self.data.index[i].timestamp())
                prev_close = self.data.iloc[i-1]['Close']
                ohlcv_data = window[['Open', 'High', 'Low', 'Close', 'Volume']].values

                try:
                    predicted = predict_func(ohlcv_data)
                except Exception as e:
                    self.error.emit(f"Runtime error in prediction: {str(e)}")
                    return

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

                # Emit update
                self.progress.emit(self.timeframe, ts, actual, predicted, is_success)

                # Dynamic delay based on total_count to keep the app responsive as data grows.
                # As total_count increases, we increase the pause to give the GUI thread more time
                # to process the increasingly heavy setData() calls.
                if i % 10 == 0:
                    # Gradually increase sleep: 5ms base + 2ms per 50 points
                    # This provides a more aggressive delay as data grows to keep the GUI responsive.
                    dynamic_pause = 0.005 + (total_count / 50) * 0.002
                    time.sleep(dynamic_pause)
                else:
                    time.sleep(0.0001) # Increased from 0.00001 to give a bit more breathing room

            final_rate = (success_count / total_count * 100) if total_count > 0 else 0
            self.finished.emit(self.timeframe, final_rate)

        except Exception as e:
            self.error.emit(f"Worker Exception: {str(e)}")

# --- Main Window ---
class GoldBacktester(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gold Futures Backtester Pro")
        self.resize(1400, 900)

        self.settings = QSettings("GoldPredictive", "Backtester")
        self.data_engine = DataEngine("data/XAU_1m_data.csv")
        self.workers = {}
        self.code_history = [] # List of tuples: (timestamp, code)
        self.timeframes = ['1m', '5m', '15m', '30m', '1h', '1d']
        self.active_timeframes = []
        self.plot_components = {}

        # Plot data storage
        self.plot_data = {tf: {
            'indices': [],
            'actuals': [],
            'predicts_x': [],
            'predicts_y': [],
            'colors': [], # List of brush colors
            'successes': [], # List of bools
            'success_count': 0
        } for tf in self.timeframes}

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
        self.settings.setValue("active_tab", self.tabs.currentIndex())
        self.settings.setValue("current_code", self.code_editor.toPlainText())

        # Save checkboxes
        tf_states = {tf: cb.isChecked() for tf, cb in self.tf_checks.items()}
        self.settings.setValue("tf_states", tf_states)

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

        active_tab = self.settings.value("active_tab", 0)
        self.tabs.setCurrentIndex(int(active_tab))

        # Restore checkboxes
        tf_states = self.settings.value("tf_states", {})
        if tf_states:
            for tf, checked in tf_states.items():
                if tf in self.tf_checks:
                    # QSettings returns strings for bools sometimes depending on platform
                    is_checked = str(checked).lower() == 'true'
                    self.tf_checks[tf].setChecked(is_checked)

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
        tf_group = QGroupBox("Select Timeframes")
        tf_vbox = QVBoxLayout()
        self.tf_checks = {}
        for tf in self.timeframes:
            cb = QCheckBox(tf)
            if tf == '1h': cb.setChecked(True) # Default
            self.tf_checks[tf] = cb
            tf_vbox.addWidget(cb)
        tf_group.setLayout(tf_vbox)
        sidebar_layout.addWidget(tf_group)

        # Thresholds
        thresh_group = QGroupBox("Success Thresholds")
        thresh_layout = QVBoxLayout()

        # Up Thresh Row
        up_hbox = QHBoxLayout()
        self.up_toggle = QCheckBox("Up Move (<=):")
        self.up_toggle.setChecked(True)
        self.up_thresh = QLineEdit("10.0")
        up_hbox.addWidget(self.up_toggle)
        up_hbox.addWidget(self.up_thresh)

        # Down Thresh Row
        down_hbox = QHBoxLayout()
        self.down_toggle = QCheckBox("Down Move (<=):")
        self.down_toggle.setChecked(True)
        self.down_thresh = QLineEdit("2.0")
        down_hbox.addWidget(self.down_toggle)
        down_hbox.addWidget(self.down_thresh)

        # Abort Thresh Row
        abort_hbox = QHBoxLayout()
        abort_label = QLabel("Auto-Abort (>):")
        self.abort_thresh = QLineEdit("100.0")
        abort_hbox.addWidget(abort_label)
        abort_hbox.addWidget(self.abort_thresh)

        thresh_layout.addLayout(up_hbox)
        thresh_layout.addLayout(down_hbox)
        thresh_layout.addLayout(abort_hbox)
        thresh_group.setLayout(thresh_layout)
        sidebar_layout.addWidget(thresh_group)

        # History
        hist_group = QGroupBox("Code History")
        hist_vbox = QVBoxLayout()
        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self.on_history_select)
        self.history_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.history_list.customContextMenuRequested.connect(self.on_history_context_menu)
        hist_vbox.addWidget(self.history_list)
        hist_group.setLayout(hist_vbox)
        sidebar_layout.addWidget(hist_group)

        sidebar_layout.addStretch()

        # --- Main Area ---
        main_area = QWidget()
        main_area_layout = QVBoxLayout(main_area)

        # Tab Widget for Charts
        self.tabs = QTabWidget()
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
            rate_label.setStyleSheet("font-size: 14pt; color: #00ff00;")
            rate_label.setAlignment(Qt.AlignRight)

            tab_layout.addWidget(rate_label)
            tab_layout.addWidget(pw)

            self.plot_widgets[tf] = pw
            self.rate_labels[tf] = rate_label
            self.tabs.addTab(tab, tf)

            # Lines
            self.plot_items[tf] = pw.plot(pen=pg.mkPen(color='#3498db', width=1.5), name="Actual")
            self.predict_items[tf] = pw.plot(pen=None, symbol='o', symbolSize=6, symbolBrush='#e74c3c', name="Predicted")

            # Create crosshair lines
            vLine = pg.InfiniteLine(angle=90, movable=False, pen='#666')
            hLine = pg.InfiniteLine(angle=0, movable=False, pen='#666')
            pw.addItem(vLine, ignoreBounds=True)
            pw.addItem(hLine, ignoreBounds=True)

            # Text item for tooltip-like display
            label = pg.TextItem(anchor=(0, 1), color='#fff', fill='#333', border='#555')
            label.setZValue(100)
            pw.addItem(label, ignoreBounds=True)

            # Store components for hover logic
            self.plot_components[tf] = {
                'vLine': vLine,
                'hLine': hLine,
                'label': label,
                'vb': pw.getViewBox()
            }

            # Connect hover signal using a lambda to pass the timeframe
            pw.scene().sigMouseMoved.connect(lambda pos, t=tf: self.on_mouse_moved(pos, t))

        main_area_layout.addWidget(self.tabs, stretch=2)

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
            QTabWidget::pane { border: 1px solid #333; }
            QTabBar::tab {
                background: #222;
                padding: 10px;
                border: 1px solid #333;
            }
            QTabBar::tab:selected {
                background: #333;
            }
            QListWidget {
                background-color: #121212;
                border: 1px solid #333;
            }
        """)

    def load_initial_code(self):
        try:
            with open("ai_studio_code_1_10.py", "r") as f:
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

    @Slot()
    def on_start_backtest(self):
        if not self.data_engine.df is not None:
            self.on_load_data()

        self.active_timeframes = [tf for tf, cb in self.tf_checks.items() if cb.isChecked()]
        if not self.active_timeframes:
            QMessageBox.warning(self, "Warning", "Select at least one timeframe.")
            return

        # Reset plots
        for tf in self.active_timeframes:
            self.plot_data[tf] = {
                'indices': [],
                'actuals': [],
                'predicts_x': [],
                'predicts_y': [],
                'colors': [],
                'successes': [],
                'success_count': 0
            }
            self.plot_items[tf].setData([], [])
            self.predict_items[tf].setData([], [])
            self.rate_labels[tf].setText("Success Rate: 0.00%")

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

        for tf in self.active_timeframes:
            df_tf = self.data_engine.get_resampled_data(tf)

            # Filter by date range
            df_tf = df_tf[(df_tf.index >= start_dt) & (df_tf.index <= end_dt)]

            if len(df_tf) < 11:
                self.rate_labels[tf].setText("Insufficient Data")
                continue

            worker = BacktestWorker(df_tf, code, tf, thresholds, toggles, abort_range)
            worker.progress.connect(self.update_plot)
            worker.finished.connect(self.on_worker_finished)
            worker.error.connect(self.on_worker_error)
            worker.aborted.connect(self.on_worker_aborted)

            self.workers[tf] = worker
            worker.start()

        self.backtest_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    @Slot(str, float, float, float, bool)
    def update_plot(self, tf, ts, actual, predicted, is_success):
        d = self.plot_data[tf]
        d['indices'].append(ts)
        d['actuals'].append(actual)
        d['predicts_x'].append(ts)
        d['predicts_y'].append(predicted)
        d['successes'].append(is_success)

        # Color code: Green for success, Red for failure
        color = '#27ae60' if is_success else '#e74c3c'
        d['colors'].append(pg.mkBrush(color))

        if is_success:
            d['success_count'] += 1

        if len(d['indices']) % 10 == 0: # Update every 10 points for smoother UI
            self.plot_items[tf].setData(d['indices'], d['actuals'])
            # Pass a copy of the colors list to avoid pyqtgraph internal size mismatch crashes
            self.predict_items[tf].setData(
                x=d['predicts_x'],
                y=d['predicts_y'],
                symbolBrush=list(d['colors'])
            )

            rate = (d['success_count'] / len(d['indices']) * 100)
            self.rate_labels[tf].setText(f"Success Rate: {rate:.2f}%")

    @Slot(str, float)
    def on_worker_finished(self, tf, rate):
        self.rate_labels[tf].setText(f"FINAL Success Rate: {rate:.2f}%")
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

    @Slot()
    def on_stop_backtest(self):
        for tf, worker in self.workers.items():
            worker.stop()
            worker.wait()
        self.workers.clear()
        self.backtest_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_mouse_moved(self, pos, tf):
        """
        Handles mouse hover to show tooltips with Actual vs Predicted values.
        """
        if tf not in self.plot_data or not self.plot_data[tf]['indices']:
            return

        components = self.plot_components.get(tf)
        if not components:
            return

        vb = components['vb']
        if vb.sceneBoundingRect().contains(pos):
            mousePoint = vb.mapSceneToView(pos)
            timestamp = mousePoint.x()

            # Find the closest index in our data
            indices = self.plot_data[tf]['indices']
            actuals = self.plot_data[tf]['actuals']
            predicts_y = self.plot_data[tf]['predicts_y']
            successes = self.plot_data[tf]['successes']

            if not indices:
                return

            # Find closest timestamp using binary search and distance comparison
            import bisect
            idx_pos = bisect.bisect_left(indices, timestamp)

            # Determine the closest index (left or right) to the cursor
            if idx_pos == 0:
                closest_idx = 0
            elif idx_pos == len(indices):
                closest_idx = len(indices) - 1
            else:
                # Compare distance to the points on either side
                left_dist = abs(timestamp - indices[idx_pos - 1])
                right_dist = abs(indices[idx_pos] - timestamp)
                if left_dist < right_dist:
                    closest_idx = idx_pos - 1
                else:
                    closest_idx = idx_pos

            if 0 <= closest_idx < len(indices):
                actual_ts = indices[closest_idx]
                actual_val = actuals[closest_idx]
                pred_val = predicts_y[closest_idx]
                is_success = successes[closest_idx]

                # Calculate changes from previous actual
                prev_actual = actuals[closest_idx - 1] if closest_idx > 0 else actual_val
                pred_change = pred_val - prev_actual
                actual_change = actual_val - prev_actual
                diff = pred_val - actual_val

                # Colors and symbols
                def get_color(val):
                    return "#27ae60" if val >= 0 else "#e74c3c"

                status_icon = "<span style='color: #27ae60'>✔</span>" if is_success else "<span style='color: #e74c3c'>✘</span>"

                # Update crosshair to follow cursor
                components['vLine'].setPos(mousePoint.x())
                components['hLine'].setPos(mousePoint.y())

                # Update tooltip text
                time_str = pd.to_datetime(actual_ts, unit='s').strftime('%Y-%m-%d %H:%M')

                text = f"<span style='color: white'>{time_str}</span><br>"

                # Prediction line
                text += f"<span style='color: white'>P: {pred_val:.2f} </span>"
                text += f"<span style='color: {get_color(pred_change)}'>({pred_change:+.2f})</span><br>"

                # Actual line
                text += f"<span style='color: white'>A: {actual_val:.2f} </span>"
                text += f"<span style='color: {get_color(actual_change)}'>({actual_change:+.2f})</span><br>"

                # Diff and Status
                text += f"<span style='color: white'>≅: </span>"
                text += f"<span style='color: {get_color(diff)}'>{diff:+.2f} </span>{status_icon}"

                components['label'].setHtml(text)
                # Attach tooltip to cursor
                components['label'].setPos(mousePoint.x(), mousePoint.y())

                components['vLine'].show()
                components['hLine'].show()
                components['label'].show()
            else:
                components['vLine'].hide()
                components['hLine'].hide()
                components['label'].hide()
        else:
            components['vLine'].hide()
            components['hLine'].hide()
            components['label'].hide()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GoldBacktester()
    window.show()
    sys.exit(app.exec())
