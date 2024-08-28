# API Reference

## Overview

This document provides detailed information about the APIs used in the Multi-Modal Sentiment Analysis for Crypto Market Prediction project. It covers both external APIs that we integrate with and internal APIs that we expose for data retrieval and predictions.

## External APIs

### 1. Twitter API

- **Base URL**: `https://api.twitter.com/2`
- **Authentication**: OAuth 1.0a
- **Endpoints Used**:
  - `GET /tweets/search/recent`: Search for recent tweets
  - `GET /users/:id/tweets`: Retrieve tweets from a specific user

For detailed documentation, refer to the [Twitter API Documentation](https://developer.twitter.com/en/docs/twitter-api).

### 2. Reddit API

- **Base URL**: `https://oauth.reddit.com`
- **Authentication**: OAuth 2.0
- **Endpoints Used**:
  - `GET /r/{subreddit}/new`: Retrieve new posts from a subreddit
  - `GET /search`: Search for posts across Reddit

For detailed documentation, refer to the [Reddit API Documentation](https://www.reddit.com/dev/api/).

### 3. Cryptocurrency Exchange API (e.g., Binance)

- **Base URL**: `https://api.binance.com`
- **Authentication**: API Key and Secret
- **Endpoints Used**:
  - `GET /api/v3/klines`: Retrieve candlestick data
  - `GET /api/v3/ticker/price`: Get latest price for a symbol

For detailed documentation, refer to the [Binance API Documentation](https://binance-docs.github.io/apidocs/).

## Internal APIs

### 1. Sentiment Analysis API

#### Analyze Sentiment

- **Endpoint**: `POST /api/v1/analyze_sentiment`
- **Description**: Analyze the sentiment of given text
- **Request Body**:
  ```json
  {
    "text": "string",
    "language": "string" (optional)
  }
  ```
- **Response**:
  ```json
  {
    "sentiment_score": float,
    "sentiment_label": string,
    "confidence": float
  }
  ```

### 2. Prediction API

#### Get Price Prediction

- **Endpoint**: `GET /api/v1/predict`
- **Description**: Get price prediction for a given cryptocurrency
- **Query Parameters**:
  - `symbol`: string (e.g., "BTC/USD")
  - `timeframe`: string (e.g., "1h", "4h", "1d")
- **Response**:
  ```json
  {
    "symbol": string,
    "predicted_price": float,
    "confidence_interval": {
      "lower": float,
      "upper": float
    },
    "timestamp": string (ISO format)
  }
  ```

### 3. Historical Data API

#### Get Historical Sentiment

- **Endpoint**: `GET /api/v1/historical_sentiment`
- **Description**: Retrieve historical sentiment data for a symbol
- **Query Parameters**:
  - `symbol`: string
  - `start_date`: string (ISO format)
  - `end_date`: string (ISO format)
- **Response**:
  ```json
  {
    "symbol": string,
    "data": [
      {
        "timestamp": string (ISO format),
        "sentiment_score": float
      }
    ]
  }
  ```

## Rate Limits

- Twitter API: 450 requests per 15-minute window
- Reddit API: 60 requests per minute
- Binance API: 1200 requests per minute
- Internal APIs: 100 requests per minute per IP address

## Error Handling

All APIs use standard HTTP status codes. In case of an error, the response body will contain an error message:

```json
{
  "error": string,
  "error_code": integer
}
```

For any questions or issues regarding the APIs, please contact the API Support team.