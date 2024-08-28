#!/bin/bash

set -e

# Function to check if a command was successful
check_status() {
    if [ $? -eq 0 ]; then
        echo "✅ $1 completed successfully"
    else
        echo "❌ $1 failed"
        exit 1
    fi
}

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
check_status "Installing requirements"

# Run fine-tuning of sentiment model
python fine_tune_sentiment.py
check_status "Fine-tuning sentiment model"

# Collect sentiment data
python collect_sentiment.py
check_status "Collecting sentiment data"

# Train the main model
python train_model.py
check_status "Training main model"

# Start the prediction process
python predict.py
check_status "Starting prediction process"

echo "🎉 All processes completed successfully!"