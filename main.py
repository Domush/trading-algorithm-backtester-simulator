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
                             QListWidget, QSplitter, QMessageBox, QCheckBox)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QFont, QColor

from data_engine import DataEngine
from highlighter import PygmentsHighlighter

# --- Worker Thread ---
class BacktestWorker(QThread):
    # Signals: timeframe, index, actual, predicted, is_success
    progress = Signal(str, int, float, float, bool)
    finished = Signal(str, float)
    error = Signal(str)
    aborted = Signal(str, str)

    def __init__(self, data, predict_code, timeframe, thresholds, abort_range):
        super().__init__()
        self.data = data
        self.predict_code = predict_code
        self.timeframe = timeframe
        self.thresholds = thresholds
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
                    if diff <= self.thresholds['up']:
                        is_success = True
                else: # Down
                    if diff <= self.thresholds['down']:
                        is_success = True

                if is_success:
                    success_count += 1
                total_count += 1

                # Emit update
                self.progress.emit(self.timeframe, i, actual, predicted, is_success)

                # Small delay to keep the app responsive and prevent CPU 100% saturation
                # This allows the GUI thread to process signals more smoothly.
                if i % 10 == 0:
                    time.sleep(0.001)
                else:
                    time.sleep(0.00001)

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

        self.data_engine = DataEngine("data/XAU_1m_data.csv")
        self.workers = {}
        self.code_history = []
        self.timeframes = ['1m', '5m', '15m', '30m', '1h', '1d']
        self.active_timeframes = []

        # Plot data storage
        self.plot_data = {tf: {'indices': [], 'actuals': [], 'predicts_x': [], 'predicts_y': []} for tf in self.timeframes}

        self.init_ui()
        self.apply_dark_theme()
        self.load_initial_code()

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
        data_vbox.addWidget(self.load_btn)
        data_vbox.addWidget(self.convert_btn)
        data_group.setLayout(data_vbox)
        sidebar_layout.addWidget(data_group)

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
        thresh_form = QFormLayout()
        self.up_thresh = QLineEdit("10.0")
        self.down_thresh = QLineEdit("2.0")
        self.abort_thresh = QLineEdit("100.0")
        thresh_form.addRow("Up Move (<=):", self.up_thresh)
        thresh_form.addRow("Down Move (<=):", self.down_thresh)
        thresh_form.addRow("Auto-Abort (>):", self.abort_thresh)
        thresh_group.setLayout(thresh_form)
        sidebar_layout.addWidget(thresh_group)

        # History
        hist_group = QGroupBox("Code History")
        hist_vbox = QVBoxLayout()
        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self.on_history_select)
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

            # Plot
            pw = pg.PlotWidget(title=f"Gold Futures - {tf}")
            pw.setBackground('#121212')
            pw.showGrid(x=True, y=True, alpha=0.3)

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
            self.plot_items[tf] = pw.plot(pen=pg.mkPen(color='#3498db', width=1.5))
            self.predict_items[tf] = pw.plot(pen=None, symbol='o', symbolSize=4, symbolBrush='#e74c3c')

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
        try:
            self.data_engine.load_data()
            QMessageBox.information(self, "Success", "Data loaded successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")

    @Slot()
    def on_convert_feather(self):
        try:
            self.data_engine.convert_to_feather()
            QMessageBox.information(self, "Success", "Data converted to Feather format!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Conversion failed: {str(e)}")

    @Slot()
    def on_update_code(self):
        code = self.code_editor.toPlainText()
        try:
            compile(code, '<string>', 'exec')
            self.backtest_btn.setEnabled(True)
            self.update_btn.setStyleSheet("background-color: #27ae60;")

            # Save to history if changed
            if not self.code_history or self.code_history[-1] != code:
                self.code_history.append(code)
                self.history_list.addItem(f"Revision {len(self.code_history)} - {time.strftime('%H:%M:%S')}")
        except Exception as e:
            self.backtest_btn.setEnabled(False)
            self.update_btn.setStyleSheet("background-color: #c0392b;")
            QMessageBox.warning(self, "Syntax Error", str(e))

    def on_history_select(self, item):
        idx = self.history_list.row(item)
        if 0 <= idx < len(self.code_history):
            self.code_editor.setPlainText(self.code_history[idx])
            self.backtest_btn.setEnabled(False)
            self.update_btn.setStyleSheet("")

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
            self.plot_data[tf] = {'indices': [], 'actuals': [], 'predicts_x': [], 'predicts_y': []}
            self.plot_items[tf].setData([], [])
            self.predict_items[tf].setData([], [])
            self.rate_labels[tf].setText("Success Rate: 0.00%")

        code = self.code_editor.toPlainText()
        thresholds = {
            'up': float(self.up_thresh.text()),
            'down': float(self.down_thresh.text())
        }
        abort_range = float(self.abort_thresh.text())

        for tf in self.active_timeframes:
            df_tf = self.data_engine.get_resampled_data(tf)
            worker = BacktestWorker(df_tf, code, tf, thresholds, abort_range)
            worker.progress.connect(self.update_plot)
            worker.finished.connect(self.on_worker_finished)
            worker.error.connect(self.on_worker_error)
            worker.aborted.connect(self.on_worker_aborted)

            self.workers[tf] = worker
            worker.start()

        self.backtest_btn.setEnabled(False)

    @Slot(str, int, float, float, bool)
    def update_plot(self, tf, idx, actual, predicted, is_success):
        d = self.plot_data[tf]
        d['indices'].append(idx)
        d['actuals'].append(actual)
        d['predicts_x'].append(idx)
        d['predicts_y'].append(predicted)

        # Batch updates for performance? pyqtgraph is fast but let's see.
        # Requirement: "don't redraw the whole graph every step. Use plotDataItem.setData() to append only the new points"
        # Actually setData() replaces the data. To append we'd need something else, but setData with full array is usually fine.
        # If it's slow, we update every N steps.

        if len(d['indices']) % 10 == 0: # Update every 10 points for smoother UI
            self.plot_items[tf].setData(d['indices'], d['actuals'])
            self.predict_items[tf].setData(d['predicts_x'], d['predicts_y'])

            # Calculate success rate so far
            # We'd need to track success count too. Let's simplify and just show it at end or track it.
            # Let's track it in plot_data
            if 'success_count' not in d: d['success_count'] = 0
            if is_success: d['success_count'] += 1

            rate = (d['success_count'] / len(d['indices']) * 100)
            self.rate_labels[tf].setText(f"Success Rate: {rate:.2f}%")

    @Slot(str, float)
    def on_worker_finished(self, tf, rate):
        self.rate_labels[tf].setText(f"FINAL Success Rate: {rate:.2f}%")
        if tf in self.workers:
            del self.workers[tf]
        if not self.workers:
            self.backtest_btn.setEnabled(True)

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

    @Slot()
    def on_stop_backtest(self):
        for tf, worker in self.workers.items():
            worker.stop()
            worker.wait()
        self.workers.clear()
        self.backtest_btn.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GoldBacktester()
    window.show()
    sys.exit(app.exec())
