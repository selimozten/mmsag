import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from utils.api_client import APIClient

class MarketDataLoader:
    def __init__(self, config):
        self.config = config
        self.api_client = self._authenticate()
        self.logger = logging.getLogger(__name__)

    def _authenticate(self):
        try:
            with open('credentials.yml', 'r') as file:
                creds = yaml.safe_load(file)['exchange']
            return APIClient('https://api.binance.com', creds['api_key'])
        except Exception as e:
            self.logger.error(f"Failed to authenticate with exchange API: {e}")
            raise

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        try:
            params = {
                'symbol': symbol,
                'interval': timeframe,
                'startTime': int(since.timestamp() * 1000),
                'limit': limit
            }
            response = self.api_client.make_request('api/v3/klines', params=params)
            
            df = pd.DataFrame(response, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.astype(float)
            return df[['open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV data for {symbol}: {e}")
            return pd.DataFrame()

    def preprocess_data(self, df):
        # Calculate additional features
        df['MA7'] = df['close'].rolling(window=7).mean()
        df['MA30'] = df['close'].rolling(window=30).mean()
        df['RSI'] = self.calculate_rsi(df['close'])
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = self.calculate_macd(df['close'])
        df['ATR'] = self.calculate_atr(df)
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = self.calculate_bollinger_bands(df['close'])

        # Normalize the data
        for column in df.columns:
            if column != 'timestamp':
                df[column] = (df[column] - df[column].mean()) / df[column].std()

        return df.dropna()

    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices, fast_period=12, slow_period=26, signal_period=9):
        exp1 = prices.ewm(span=fast_period, adjust=False).mean()
        exp2 = prices.ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist

    def calculate_atr(self, df, period=14):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()

    def calculate_bollinger_bands(self, prices, period=20, num_std=2):
        rolling_mean = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, rolling_mean, lower_band

    def load_training_data(self):
        X_market = []
        X_sentiment = []  # Placeholder for sentiment data
        y = []

        for symbol in self.config['symbols']:
            df = self.fetch_ohlcv(symbol, self.config['timeframe'], 
                                  datetime.utcnow() - timedelta(days=self.config['lookback_period']),
                                  self.config['lookback_period'] * 24)  # Assuming hourly data
            df = self.preprocess_data(df)

            for i in range(len(df) - self.config['lookback_period']):
                X_market.append(df.iloc[i:i+self.config['lookback_period']].values)
                y.append(df.iloc[i+self.config['lookback_period']]['close'])

        return np.array(X_market), np.array(X_sentiment), np.array(y)

    def get_latest_data(self, symbol):
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=self.config['lookback_period'])
        df = self.fetch_ohlcv(symbol, self.config['timeframe'], start_time, self.config['lookback_period'])

        if df.empty:
            self.logger.warning(f"No data retrieved for {symbol}. Returning None.")
            return None

        df = self.preprocess_data(df)

        if len(df) < self.config['lookback_period']:
            self.logger.warning(f"Insufficient data for {symbol}. Expected {self.config['lookback_period']} rows, got {len(df)}.")
            return None

        return df.iloc[-self.config['lookback_period']:].values.reshape(1, self.config['lookback_period'], -1)