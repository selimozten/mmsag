# Multi-Modal Sentiment Analysis for Crypto Market Prediction

## Overview

This project implements a sophisticated sentiment analysis system that combines market data with social media sentiment to predict cryptocurrency price movements. It's designed to give our exchange a competitive edge by incorporating real-time public sentiment into our trading strategies.

## Features

- Integration with Twitter and Reddit APIs for sentiment data collection
- Advanced NLP model for crypto-specific sentiment analysis
- Multi-input neural network combining market data and sentiment scores
- Real-time prediction system for multiple cryptocurrencies

## Prerequisites

- Python 3.8+
- TensorFlow 2.4+
- transformers
- tweepy
- praw
- pandas
- numpy
- ccxt

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/selimozten/mmsag.git
   ```

2. Navigate to the project directory:
   ```
   cd mmsag
   ```

3. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

5. Set up your API credentials:
   - Copy `credentials.example.yml` to `credentials.yml`
   - Fill in your Twitter, Reddit, and exchange API credentials

## Configuration

Edit `config.yml` to customize the project settings:
- `symbols`: List of cryptocurrency pairs to analyze
- `timeframe`: Data timeframe for market analysis
- `sentiment_window`: Time window for sentiment analysis
- `model_params`: Neural network architecture parameters

## Usage

1. To run the sentiment collection and analysis:
   ```
   python collect_sentiment.py
   ```

2. To train the prediction model:
   ```
   python train_model.py
   ```

3. To make real-time predictions:
   ```
   python predict.py
   ```

## Project Structure

- `collect_sentiment.py`: Script for collecting and analyzing sentiment data
- `train_model.py`: Script for training the multi-input neural network
- `predict.py`: Script for making real-time predictions
- `models/`: Directory containing model definitions
- `data_processing/`: Modules for data fetching and preprocessing
- `sentiment_analysis/`: Modules for sentiment analysis
- `utils/`: Utility functions and helpers
- `config.yml`: Project configuration file
- `credentials.yml`: API credentials (do not commit this file)
- `tests/`: Directory containing unit and integration tests

## Extending the Project

### Adding New Data Sources

1. Create a new module in `data_processing/` for the new source
2. Implement the required API calls and data parsing
3. Update `collect_sentiment.py` to incorporate the new source
4. Modify the model in `models/multi_input_nn.py` if necessary

### Improving the Sentiment Analysis

1. Fine-tune the sentiment model on crypto-specific data:
   ```
   python fine_tune_sentiment.py
   ```
2. Experiment with different NLP models in `sentiment_analysis/model.py`

## Testing

Run the test suite with:
```
python -m pytest tests/
```

Ensure all tests pass before deploying any changes to production.

## Monitoring and Logging

- Logs are stored in the `logs/` directory
- Use the `logging` module for consistent log formatting
- Monitor model performance using TensorBoard:
  ```
  tensorboard --logdir=./logs/tensorboard
  ```

## Security Notes

- Never commit `credentials.yml` or any file containing API keys
- Rotate API keys regularly and update them in our secure key management system
- Ensure all data processing follows our data protection guidelines

## Internal Resources

- For a detailed explanation of the sentiment analysis algorithm, see `docs/sentiment_algorithm.md`
- For guidelines on model deployment, refer to our MLOps playbook in the company wiki

## Support

For issues or feature requests, please create an issue in this repository or contact the Data Science team directly.
