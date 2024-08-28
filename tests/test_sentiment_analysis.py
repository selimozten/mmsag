import unittest
from unittest.mock import patch, MagicMock
from sentiment_analysis.model import SentimentAnalyzer

class TestSentimentAnalyzer(unittest.TestCase):
    @patch('sentiment_analysis.model.AutoModelForSequenceClassification')
    @patch('sentiment_analysis.model.AutoTokenizer')
    def setUp(self, mock_tokenizer, mock_model):
        self.analyzer = SentimentAnalyzer()
        self.mock_tokenizer = mock_tokenizer
        self.mock_model = mock_model

    def test_analyze_batch(self):
        texts = ['I love crypto', 'Bitcoin is crashing']
        self.mock_tokenizer.return_value.return_value = {'input_ids': MagicMock(), 'attention_mask': MagicMock()}
        self.mock_model.return_value.return_value = MagicMock(logits=MagicMock())
        
        sentiments = self.analyzer.analyze_batch(texts)
        
        self.assertEqual(len(sentiments), 2)
        self.assertTrue(all(0 <= s <= 1 for s in sentiments))