import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from models.multi_input_nn import MultiInputNN
from data_processing.market_data import MarketDataLoader
from sentiment_analysis.model import SentimentAnalyzer
from utils.helpers import load_config, prepare_prediction_input

def load_latest_data(market_data_loader, sentiment_analyzer, symbol):
    market_data = market_data_loader.get_latest_data(symbol)
    sentiment_data = sentiment_analyzer.get_latest_sentiment(symbol, market_data_loader)
    return market_data, sentiment_data

def make_prediction(model, market_data, sentiment_data):
    X_market, X_sentiment = prepare_prediction_input(market_data, sentiment_data)
    return model.predict(X_market, X_sentiment)[0][0]

def plot_predictions(predictions_df):
    plt.figure(figsize=(12, 6))
    for symbol in predictions_df['symbol'].unique():
        symbol_data = predictions_df[predictions_df['symbol'] == symbol]
        plt.plot(symbol_data['timestamp'], symbol_data['prediction'], label=symbol)
    plt.title('Price Predictions Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Predicted Price')
    plt.legend()
    return plt

def main():
    st.title('Crypto Market Prediction Dashboard')

    config = load_config()
    market_data_loader = MarketDataLoader(config['market_data'])
    sentiment_analyzer = SentimentAnalyzer()
    model = MultiInputNN.load('models/trained_multi_input_model.h5', config['model_params'])

    # Sidebar for user input
    st.sidebar.header('Settings')
    selected_symbols = st.sidebar.multiselect('Select cryptocurrencies', config['symbols'], default=config['symbols'][0])
    prediction_days = st.sidebar.slider('Prediction days', 1, 30, 7)

    # Make predictions
    predictions = []
    for symbol in selected_symbols:
        market_data, sentiment_data = load_latest_data(market_data_loader, sentiment_analyzer, symbol)
        
        for i in range(prediction_days):
            timestamp = datetime.now() + timedelta(days=i)
            prediction = make_prediction(model, market_data, sentiment_data)
            predictions.append({'symbol': symbol, 'timestamp': timestamp, 'prediction': prediction})
            
            # Update market_data and sentiment_data for next prediction
            # This is a simplification; in a real scenario, you'd need a more sophisticated method
            market_data = market_data[:, 1:, :]
            market_data = np.concatenate([market_data, prediction.reshape(1, 1, -1)], axis=1)

    predictions_df = pd.DataFrame(predictions)

    # Display predictions
    st.header('Price Predictions')
    st.dataframe(predictions_df)

    # Plot predictions
    st.header('Prediction Chart')
    fig = plot_predictions(predictions_df)
    st.pyplot(fig)

    # Display additional metrics or analyses
    st.header('Additional Metrics')
    for symbol in selected_symbols:
        symbol_data = predictions_df[predictions_df['symbol'] == symbol]
        st.subheader(f'{symbol} Metrics')
        st.write(f"Mean predicted price: {symbol_data['prediction'].mean():.2f}")
        st.write(f"Predicted price change: {(symbol_data['prediction'].iloc[-1] - symbol_data['prediction'].iloc[0]) / symbol_data['prediction'].iloc[0] * 100:.2f}%")

if __name__ == '__main__':
    main()