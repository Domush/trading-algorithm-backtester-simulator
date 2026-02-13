import pandas as pd
import os

class DataEngine:
    def __init__(self, csv_path=None):
        self.csv_path = csv_path
        self.feather_path = csv_path.replace('.csv', '.feather') if csv_path else None
        self.df = None

    def set_csv_path(self, csv_path):
        self.csv_path = csv_path
        self.feather_path = csv_path.replace('.csv', '.feather') if csv_path else None

    def load_data(self, force_csv=False):
        if not self.csv_path:
            raise ValueError("No CSV path provided.")

        if not force_csv and self.feather_path and os.path.exists(self.feather_path):
            print(f"Loading data from {self.feather_path}...")
            self.df = pd.read_feather(self.feather_path)
            if 'Date' in self.df.columns:
                self.df.set_index('Date', inplace=True)
        else:
            print(f"Loading data from {self.csv_path}...")
            self.df = pd.read_csv(self.csv_path)
            # Convert Date to datetime and set as index
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df.set_index('Date', inplace=True)
            # Use float32 for memory efficiency as per Pro Tips
            float_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            self.df[float_cols] = self.df[float_cols].astype('float32')

        return self.df

    def convert_to_feather(self):
        if self.df is None:
            self.load_data(force_csv=True)
        print(f"Converting to {self.feather_path}...")
        # reset_index() because feather doesn't support datetime index directly in some versions
        self.df.reset_index().to_feather(self.feather_path)
        print("Conversion complete.")

    def get_resampled_data(self, timeframe):
        """
        Resamples data to given timeframe (e.g., '5min', '15min', '30min', '1h', '1d')
        """
        if self.df is None:
            self.load_data()

        # Map human-readable timeframes to pandas resampling strings
        tf_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1h',
            '1d': '1d'
        }

        tf = tf_map.get(timeframe, timeframe)

        if tf == '1min':
            return self.df

        logic = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
        return self.df.resample(tf).apply(logic).dropna()
