import unittest
from unittest.mock import patch
from data_processing.reddit_data import RedditCollector

class TestRedditCollector(unittest.TestCase):
    def setUp(self):
        self.config = {'subreddits': ['CryptoCurrency'], 'post_limit': 100, 'sentiment': {'window': 24}}
        self.collector = RedditCollector(self.config)

    @patch('data_processing.reddit_data.APIClient')
    def test_collect_posts(self, mock_api_client):
        mock_response = {'data': {'children': [{'data': {'title': 'Test post', 'selftext': 'Content', 'created_utc': 1630000000}}]}}
        mock_api_client.return_value.make_request.return_value = mock_response
        
        posts = self.collector.collect_posts('BTC', 24)
        
        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0], 'Test post Content')