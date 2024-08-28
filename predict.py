import logging
import pandas as pd
from models.multi_input_nn import MultiInputNN
from data_processing.market_data import MarketDataLoader
from sentiment_analysis.model import SentimentAnalyzer
from utils.helpers import setup_logging, load_config, prepare_prediction_input, wait_for_next_cycle, save_predictions, plot_predictions
from datetime import datetime
import time

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    config = load_config()

    logger.info("Starting prediction process")

    # Load the trained model
    model = MultiInputNN.load('models/trained_multi_input_model.h5', config['model_params'])

    # Initialize data collectors and sentiment analyzer
    market_data_loader = MarketDataLoader(config['market_data'])
    sentiment_analyzer = SentimentAnalyzer(model_path='models/fine_tuned_sentiment_model')

    all_predictions = []

    while True:
        predictions = []
        timestamp = datetime.utcnow()

        for symbol in config['symbols']:
            # Fetch latest market data and sentiment
            latest_market_data = market_data_loader.get_latest_data(symbol)
            latest_sentiment = sentiment_analyzer.get_latest_sentiment(symbol, market_data_loader)

            if latest_market_data is None:
                logger.warning(f"Skipping prediction for {symbol} due to missing market data")
                continue

            # Prepare input for prediction
            X_market, X_sentiment = prepare_prediction_input(latest_market_data, latest_sentiment)

            # Make prediction
            prediction = model.predict([X_market, X_sentiment])
            predictions.append((symbol, prediction[0][0]))

            logger.info(f"Prediction for {symbol}: {prediction[0][0]:.4f}")

        # Save predictions
        save_predictions(predictions, config['symbols'], timestamp)
        all_predictions.extend([(symbol, pred, timestamp) for symbol, pred in predictions])

        # Plot predictions
        if len(all_predictions) > 0:
            plot_predictions(all_predictions)

        # Wait for the next prediction cycle
        wait_for_next_cycle(config['prediction']['interval'])

if __name__ == "__main__":
    main()