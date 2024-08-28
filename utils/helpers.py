import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import TensorBoard
import yaml
import pandas as pd

def setup_logging():
    with open('config.yml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    logging.basicConfig(level=config['logging']['level'],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename=config['logging']['file'])
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def load_config():
    with open('config.yml', 'r') as config_file:
        return yaml.safe_load(config_file)

def aggregate_sentiment(tweet_sentiments, reddit_sentiments):
    combined = np.concatenate([tweet_sentiments, reddit_sentiments])
    return np.mean(combined)

def prepare_prediction_input(market_data, sentiment_data):
    X_market = np.array(market_data).reshape(1, -1, market_data.shape[-1])
    X_sentiment = np.array(sentiment_data).reshape(1, -1, 1)
    return X_market, X_sentiment

def wait_for_next_cycle(interval):
    next_cycle = (int(time.time()) // interval + 1) * interval
    time.sleep(next_cycle - time.time())

def log_training_results(history):
    logger = logging.getLogger(__name__)
    logger.info("Training completed.")
    logger.info(f"Final loss: {history.history['loss'][-1]:.4f}")
    logger.info(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

def create_tensorboard_callback(log_dir):
    return TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

def evaluate_model(model, X_test, y_test):
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    logger = logging.getLogger(__name__)
    logger.info(f"Test Loss: {loss:.4f}")
    logger.info(f"Test MAE: {mae:.4f}")
    return loss, mae

def save_predictions(predictions, symbols, timestamp):
    logger = logging.getLogger(__name__)
    df = pd.DataFrame(predictions, columns=['symbol', 'prediction'])
    df['timestamp'] = timestamp
    df.to_csv(f'data/predictions_{timestamp.strftime("%Y%m%d_%H%M%S")}.csv', index=False)
    logger.info(f"Predictions saved to data/predictions_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv")

def plot_sentiment_results(df):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='symbol', y='sentiment', data=df)
    plt.title('Aggregated Sentiment by Symbol')
    plt.xlabel('Symbol')
    plt.ylabel('Sentiment Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/sentiment_results.png')
    plt.close()

def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/training_history.png')
    plt.close()

def plot_predictions(predictions):
    df = pd.DataFrame(predictions, columns=['symbol', 'prediction', 'timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    plt.figure(figsize=(12, 6))
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol]
        plt.plot(symbol_data['timestamp'], symbol_data['prediction'], label=symbol)
    
    plt.title('Price Predictions Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Predicted Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/predictions.png')
    plt.close()

def create_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png')
    plt.close()

def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances")
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')
    plt.close()

def create_correlation_heatmap(df):
    corr = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('plots/correlation_heatmap.png')
    plt.close()