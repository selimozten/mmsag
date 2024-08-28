import logging
import pandas as pd
from data_processing.twitter_data import TwitterCollector
from data_processing.reddit_data import RedditCollector
from sentiment_analysis.model import SentimentAnalyzer
from utils.helpers import setup_logging, load_config, aggregate_sentiment
from tqdm import tqdm

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    config = load_config()

    logger.info("Starting sentiment collection process")

    # Initialize data collectors and sentiment analyzer
    twitter_collector = TwitterCollector(config['twitter'])
    reddit_collector = RedditCollector(config['reddit'])
    sentiment_analyzer = SentimentAnalyzer(model_path='models/fine_tuned_sentiment_model')

    all_sentiments = []

    for symbol in tqdm(config['symbols'], desc="Processing symbols"):
        logger.info(f"Collecting and analyzing sentiment for {symbol}")

        # Collect data
        tweets = twitter_collector.collect_tweets(symbol, config['sentiment']['window'])
        reddit_posts = reddit_collector.collect_posts(symbol, config['sentiment']['window'])

        # Analyze sentiment
        tweet_sentiments = sentiment_analyzer.analyze_batch(tweets)
        reddit_sentiments = sentiment_analyzer.analyze_batch(reddit_posts)

        # Aggregate results
        aggregated_sentiment = aggregate_sentiment(tweet_sentiments, reddit_sentiments)

        all_sentiments.append({
            'symbol': symbol,
            'sentiment': aggregated_sentiment,
            'tweet_count': len(tweets),
            'reddit_post_count': len(reddit_posts)
        })

        logger.info(f"Sentiment analysis complete for {symbol}: {aggregated_sentiment:.4f}")

    # Save results to CSV
    df = pd.DataFrame(all_sentiments)
    df.to_csv('data/sentiment_results.csv', index=False)
    logger.info("Sentiment results saved to data/sentiment_results.csv")

    # Visualize results
    plot_sentiment_results(df)

if __name__ == "__main__":
    main()