import logging
import numpy as np
from models.multi_input_nn import MultiInputNN
from data_processing.market_data import MarketDataLoader
from utils.helpers import setup_logging, load_config, create_tensorboard_callback, evaluate_model, plot_training_history
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    config = load_config()

    logger.info("Starting model training process")

    # Initialize data loader and load data
    market_data_loader = MarketDataLoader(config['market_data'])
    X_market, X_sentiment, y = market_data_loader.load_training_data()

    # Split data into training and testing sets
    X_market_train, X_market_test, X_sentiment_train, X_sentiment_test, y_train, y_test = train_test_split(
        X_market, X_sentiment, y, test_size=0.2, random_state=42
    )

    # Initialize and compile the model
    model = MultiInputNN(config['model_params'])
    model.compile_model()

    # Create callbacks
    tensorboard_callback = create_tensorboard_callback(config['tensorboard']['log_dir'])
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6)

    # Train the model
    history = model.fit(
        [X_market_train, X_sentiment_train], y_train,
        validation_data=([X_market_test, X_sentiment_test], y_test),
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        callbacks=[tensorboard_callback, early_stopping, reduce_lr]
    )

    # Evaluate the model
    evaluate_model(model, [X_market_test, X_sentiment_test], y_test)

    # Plot training history
    plot_training_history(history)

    # Save the model
    model.save('models/trained_multi_input_model.h5')
    logger.info("Model training completed and saved.")

if __name__ == "__main__":
    main()