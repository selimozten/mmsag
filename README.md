# Multi-Modal Sentiment Analysis for Crypto Market Prediction

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [API Credentials](#api-credentials)
7. [Usage](#usage)
8. [Project Structure](#project-structure)
9. [Extending the Project](#extending-the-project)
10. [Testing](#testing)
11. [Monitoring and Logging](#monitoring-and-logging)
12. [Security Notes](#security-notes)
13. [Performance Optimization](#performance-optimization)
14. [Contribution Guidelines](#contribution-guidelines)
15. [Troubleshooting](#troubleshooting)
16. [FAQ](#faq)
17. [Internal Resources](#internal-resources)
18. [Support](#support)
19. [License](#license)

## Overview

This project implements a sophisticated sentiment analysis system that combines market data with social media sentiment to predict cryptocurrency price movements. It's designed to give our exchange a competitive edge by incorporating real-time public sentiment into our trading strategies. By leveraging advanced natural language processing techniques and deep learning models, we aim to capture market sentiment and its impact on cryptocurrency prices more accurately than traditional technical analysis alone.

## Features

- **Data Collection**:
  - Integration with Twitter API for real-time tweet collection
  - Reddit data scraping for cryptocurrency-related posts
  - Historical and real-time market data retrieval from major exchanges
- **Sentiment Analysis**:
  - Advanced NLP model fine-tuned for crypto-specific sentiment analysis
  - Multi-lingual support for global market sentiment capture
- **Market Prediction**:
  - Multi-input neural network combining market data and sentiment scores
  - Real-time prediction system for multiple cryptocurrencies
  - Customizable prediction timeframes
- **Visualization**:
  - Interactive dashboard for visualizing predictions and market trends
  - Detailed charts and metrics for in-depth analysis
- **Extensibility**:
  - Modular design for easy integration of new data sources and models
  - Configurable parameters for fine-tuning the system

## Prerequisites

- Python 3.8+
- TensorFlow 2.4+
- PyTorch 1.9+
- CUDA-capable GPU (recommended for faster training and inference)
- Access to Twitter and Reddit APIs
- Account with a supported cryptocurrency exchange (e.g., Binance)

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

5. Install additional dependencies for GPU support (if available):
   ```
   pip install tensorflow-gpu torch-gpu
   ```

## Configuration

1. Copy the example configuration file:
   ```
   cp example.config.yml config.yml
   ```

2. Edit `config.yml` to customize the project settings:
   - `symbols`: List of cryptocurrency pairs to analyze
   - `timeframe`: Data timeframe for market analysis
   - `sentiment_window`: Time window for sentiment analysis
   - `model_params`: Neural network architecture parameters
   - `training`: Parameters for model training
   - `prediction`: Settings for real-time predictions
   - `logging`: Configuration for application logs
   - `dashboard`: Settings for the Streamlit dashboard

Refer to the comments in `example.config.yml` for detailed explanations of each setting.

## API Credentials

1. Copy the example credentials file:
   ```
   cp example.credentials.yml credentials.yml
   ```

2. Edit `credentials.yml` and fill in your API credentials for:
   - Twitter API (consumer key, consumer secret, access token, access token secret)
   - Reddit API (client ID, client secret, user agent)
   - Exchange API (API key, secret key)
   - Sentiment Analysis API (if using a paid service)

**Important:** Never commit your `credentials.yml` file to version control. It's included in the `.gitignore` file to prevent accidental commits.

## Usage

1. Collect and analyze sentiment data:
   ```
   python collect_sentiment.py
   ```

2. Train the prediction model:
   ```
   python train_model.py
   ```

3. Make real-time predictions:
   ```
   python predict.py
   ```

4. Run the prediction dashboard:
   ```
   streamlit run dashboards/prediction_dashboard.py
   ```

For scheduled runs, consider using cron jobs or a task scheduler appropriate for your operating system.

## Project Structure

- `collect_sentiment.py`: Script for collecting and analyzing sentiment data
- `train_model.py`: Script for training the multi-input neural network
- `predict.py`: Script for making real-time predictions
- `models/`: Directory containing model definitions
  - `multi_input_nn.py`: Implementation of the multi-input neural network
- `data_processing/`: Modules for data fetching and preprocessing
  - `twitter_data.py`: Twitter data collection module
  - `reddit_data.py`: Reddit data collection module
  - `market_data.py`: Cryptocurrency market data retrieval module
- `sentiment_analysis/`: Modules for sentiment analysis
  - `model.py`: Sentiment analysis model implementation
- `utils/`: Utility functions and helpers
- `config.yml`: Project configuration file
- `credentials.yml`: API credentials (do not commit this file)
- `tests/`: Directory containing unit tests
- `dashboards/`: Directory containing Streamlit dashboard files
- `logs/`: Directory for log files (created at runtime)
- `example.config.yml`: Example configuration file
- `example.credentials.yml`: Example credentials file
- `.gitignore`: Specifies intentionally untracked files to ignore
- `requirements.txt`: List of Python package dependencies
- `README.md`: This file, containing project documentation

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
python -m unittest discover tests
```

Ensure all tests pass before deploying any changes to production. To run tests with coverage report:

```
coverage run -m unittest discover tests
coverage report -m
```

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
- Implement proper input validation and sanitization, especially for user inputs in the dashboard
- Regularly update dependencies to patch security vulnerabilities

## Performance Optimization

- Use batch processing for large datasets
- Implement caching mechanisms for frequently accessed data
- Optimize database queries and indexes
- Consider distributing workloads across multiple machines for scalability

## Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure your code adheres to our coding standards and includes appropriate tests.

## Troubleshooting

- Check the `logs/` directory for detailed error messages
- Ensure all API credentials are correct and have the necessary permissions
- Verify that your system meets all the prerequisites, including GPU drivers if using GPU acceleration
- For common issues, refer to the [FAQ](#faq) section

## FAQ

Q: How often should I retrain the model?
A: We recommend retraining the model weekly or when there's a significant change in market conditions.

Q: Can I use this system for high-frequency trading?
A: The current implementation is not optimized for high-frequency trading. It's designed for short to medium-term predictions.

## Internal Resources

- For a detailed explanation of the sentiment analysis algorithm, see `docs/sentiment_algorithm.md`
- For guidelines on model deployment, refer to our MLOps playbook in the company wiki
- API documentation can be found in `docs/api_reference.md`

## Support

For issues or feature requests, please create an issue in this repository or contact the Data Science team directly at ozten@inpocket.ai

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.