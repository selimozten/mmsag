from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

class SentimentAnalyzer:
    def __init__(self, model_path='models/fine_tuned_sentiment_model'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def analyze_batch(self, texts, batch_size=32):
        sentiments = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
            with torch.no_grad():
                outputs = self.model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
            sentiments.extend(scores[:, 2].tolist())  # Assuming index 2 corresponds to positive sentiment
        return np.array(sentiments)

    def get_latest_sentiment(self, symbol, data_collector):
        recent_texts = data_collector.get_recent_data(symbol)
        sentiments = self.analyze_batch(recent_texts)
        return np.mean(sentiments)

    @staticmethod
    def preprocess_text(text):
        # Add any text preprocessing steps here
        return text.lower()