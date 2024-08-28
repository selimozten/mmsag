import unittest
from unittest.mock import patch
from data_processing.twitter_data import TwitterCollector

class TestTwitterCollector(unittest.TestCase):
    def setUp(self):
        self.config = {'max_tweets': 100, 'sentiment': {'window': 24}}
        self.collector = TwitterCollector(self.config)

    @patch('data_processing.twitter_data.APIClient')
    def test_collect_tweets(self, mock_api_client):
        mock_response = {'data': [{'text': 'Test tweet'}]}
        mock_api_client.return_value.make_request.return_value = mock_response
        
        tweets = self.collector.collect_tweets('BTC', 24)
        
        self.assertEqual(len(tweets), 1)
        self.assertEqual(tweets[0], 'Test tweet')