import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
from utils.helpers import setup_logging, load_config
import torch

def load_crypto_sentiment_data(file_path):
    df = pd.read_csv(file_path)
    return Dataset.from_pandas(df)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.Tensor(logits), dim=-1)
    return {"accuracy": (predictions == torch.Tensor(labels)).float().mean().item()}

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    config = load_config()

    logger.info("Starting sentiment model fine-tuning process")

    # Load crypto-specific sentiment data
    dataset = load_crypto_sentiment_data('data/crypto_sentiment.csv')
    train_dataset, eval_dataset = train_dataset.train_test_split(test_size=0.2, seed=42).values()

    # Load pre-trained model and tokenizer
    model_name = "finiteautomata/bertweet-base-sentiment-analysis"
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Tokenize datasets
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
    )

    # Fine-tune the model
    logger.info("Fine-tuning sentiment model...")
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained('models/fine_tuned_sentiment_model')
    tokenizer.save_pretrained('models/fine_tuned_sentiment_tokenizer')

    logger.info("Fine-tuning complete. Model saved.")

if __name__ == "__main__":
    main()