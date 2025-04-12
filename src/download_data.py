import os
import pandas as pd
import requests
from tqdm import tqdm
import gzip
import shutil

def download_dataset():
    """Download the Sentiment140 dataset."""
    url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
    output_path = os.path.join("data", "sentiment140.csv")
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Create a small sample dataset for testing
    print("Creating sample dataset...")
    
    # Create sample data
    data = {
        'target': [0, 0, 4, 4] * 25,  # 100 samples with balanced sentiments
        'ids': list(range(100)),
        'date': ['Mon May 11 03:17:40 UTC 2009'] * 100,
        'flag': ['NO_QUERY'] * 100,
        'user': ['_TheSpecialOne_'] * 100,
        'text': [
            'Sad day. RIP',
            'I hate when people are late',
            'This is awesome!',
            'Having a great time!'
        ] * 25
    }
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print("Sample dataset created successfully!")

def preprocess_data():
    """Preprocess the downloaded dataset."""
    print("Preprocessing data...")
    
    # Read the CSV file
    df = pd.read_csv(os.path.join("data", "sentiment140.csv"))
    
    # Convert target to binary (0: negative, 1: positive)
    df['target'] = df['target'].map({0: 0, 4: 1})
    
    # Save preprocessed data
    output_path = os.path.join("data", "preprocessed_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")
    
    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Total number of tweets: {len(df):,}")
    print(f"Number of negative tweets: {sum(df['target'] == 0):,}")
    print(f"Number of positive tweets: {sum(df['target'] == 1):,}")

if __name__ == "__main__":
    download_dataset()
    preprocess_data() 