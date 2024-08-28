import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

class MultiInputNN:
    def __init__(self, config):
        self.config = config
        self.model = self.build_model()

    def build_model(self):
        # Market data input
        market_input = Input(shape=(self.config['market_input_dim'], self.config['market_features']), name='market_input')
        market_lstm = LSTM(self.config['lstm_units'], return_sequences=False, kernel_regularizer=l2(0.01))(market_input)
        market_lstm = BatchNormalization()(market_lstm)

        # Sentiment input
        sentiment_input = Input(shape=(self.config['sentiment_input_dim'], 1), name='sentiment_input')
        sentiment_lstm = LSTM(self.config['lstm_units'], return_sequences=False, kernel_regularizer=l2(0.01))(sentiment_input)
        sentiment_lstm = BatchNormalization()(sentiment_lstm)

        # Concatenate market and sentiment features
        combined = Concatenate()([market_lstm, sentiment_lstm])

        # Dense layers
        for i, units in enumerate(self.config['dense_units']):
            combined = Dense(units, activation='relu', kernel_regularizer=l2(0.01), name=f'dense_{i}')(combined)
            combined = BatchNormalization()(combined)
            combined = Dropout(self.config['dropout_rate'])(combined)

        # Output layer
        output = Dense(1, activation='linear', name='price_prediction')(combined)

        # Create model
        model = Model(inputs=[market_input, sentiment_input], outputs=output)

        return model

    def compile_model(self, optimizer='adam', loss='mean_squared_error'):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])

    def summary(self):
        return self.model.summary()

    def fit(self, X_market, X_sentiment, y, **kwargs):
        return self.model.fit([X_market, X_sentiment], y, **kwargs)

    def predict(self, X_market, X_sentiment):
        return self.model.predict([X_market, X_sentiment])

    def save(self, filepath):
        self.model.save(filepath)

    @classmethod
    def load(cls, filepath, config):
        instance = cls(config)
        instance.model = tf.keras.models.load_model(filepath)
        return instance

    def get_feature_importance(self):
        # This is a simplified version of feature importance
        # For more accurate results, consider using SHAP values or other model-specific techniques
        input_layer = self.model.get_layer('market_input')
        output_layer = self.model.get_layer('price_prediction')
        
        grads = tf.gradients(output_layer.output, input_layer.output)[0]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
        
        return pooled_grads.numpy()