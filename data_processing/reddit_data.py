import praw
import yaml
import logging
from datetime import datetime, timedelta
from utils.api_client import APIClient

class RedditCollector:
    def __init__(self, config):
        self.config = config
        self.api_client = self._authenticate()
        self.logger = logging.getLogger(__name__)

    def _authenticate(self):
        try:
            with open('credentials.yml', 'r') as file:
                creds = yaml.safe_load(file)['reddit']
            return APIClient('https://oauth.reddit.com', creds['access_token'])
        except Exception as e:
            self.logger.error(f"Failed to authenticate with Reddit API: {e}")
            raise

    def collect_posts(self, symbol, time_window):
        posts = []
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=time_window)

        try:
            for subreddit_name in self.config['subreddits']:
                params = {
                    'q': symbol,
                    'subreddit': subreddit_name,
                    'sort': 'new',
                    'limit': self.config['post_limit'],
                    't': 'all'
                }
                response = self.api_client.make_request('search', params=params)
                
                for post in response.get('data', {}).get('children', []):
                    post_data = post['data']
                    created_time = datetime.fromtimestamp(post_data['created_utc'])
                    if start_time <= created_time <= end_time:
                        posts.append(post_data['title'] + " " + post_data['selftext'])
                    elif created_time < start_time:
                        break

            self.logger.info(f"Collected {len(posts)} Reddit posts for {symbol}")
            return posts
        except Exception as e:
            self.logger.error(f"Error collecting Reddit posts for {symbol}: {e}")
            return []

    def get_recent_data(self, symbol):
        return self.collect_posts(symbol, self.config['sentiment']['window'])