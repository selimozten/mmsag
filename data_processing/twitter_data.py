import tweepy
import yaml
import logging
from datetime import datetime, timedelta
from utils.api_client import APIClient

class TwitterCollector:
    def __init__(self, config):
        self.config = config
        self.api_client = self._authenticate()
        self.logger = logging.getLogger(__name__)

    def _authenticate(self):
        try:
            with open('credentials.yml', 'r') as file:
                creds = yaml.safe_load(file)['twitter']
            return APIClient('https://api.twitter.com/2', creds['bearer_token'])
        except Exception as e:
            self.logger.error(f"Failed to authenticate with Twitter API: {e}")
            raise

    def collect_tweets(self, symbol, time_window):
        query = f"${symbol} OR #{symbol.lower()}"
        tweets = []
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=time_window)

        try:
            params = {
                'query': query,
                'max_results': self.config['max_tweets'],
                'start_time': start_time.isoformat() + 'Z',
                'end_time': end_time.isoformat() + 'Z',
                'tweet.fields': 'created_at,text'
            }
            response = self.api_client.make_request('tweets/search/recent', params=params)
            
            for tweet in response.get('data', []):
                tweets.append(tweet['text'])

            self.logger.info(f"Collected {len(tweets)} tweets for {symbol}")
            return tweets
        except Exception as e:
            self.logger.error(f"Error collecting tweets for {symbol}: {e}")
            return []

    def get_recent_data(self, symbol):
        return self.collect_tweets(symbol, self.config['sentiment']['window'])