import unittest
from unittest.mock import patch
import pandas as pd
from data_processing.market_data import MarketDataLoader

class TestMarketDataLoader(unittest.TestCase):
    def setUp(self):
        self.config = {'symbols': ['BTC/USD'], 'timeframe': '1h', 'lookback_period': 168}
        self.loader = MarketDataLoader(self.config)

    @patch('data_processing.market_data.APIClient')
    def test_fetch_ohlcv(self, mock_api_client):
        mock_data = [[1630000000000, '50000', '51000', '49000', '50500', '1000']]
        mock_api_client.return_value.make_request.return_value = mock_data
        
        df = self.loader.fetch_ohlcv('BTC/USD', '1h', pd.Timestamp.now(), 1)
        
        self.assertFalse(df.empty)
        self.assertEqual(df.index[0], pd.Timestamp('2021-08-26 18:40:00'))
        self.assertEqual(df['close'].values[0], 50500.0)

    def test_calculate_rsi(self):
        prices = pd.Series([10, 12, 11, 13, 14, 13, 15])
        rsi = self.loader.calculate_rsi(prices, period=5)
        self.assertAlmostEqual(rsi.iloc[-1], 70.53, places=2)