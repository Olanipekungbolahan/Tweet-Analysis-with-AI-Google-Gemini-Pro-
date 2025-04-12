import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os

def load_data():
    """Load the preprocessed dataset."""
    data_path = os.path.join("data", "preprocessed_data.csv")
    return pd.read_csv(data_path)

def analyze_sentiment_distribution(df):
    """Analyze and visualize the distribution of sentiments."""
    plt.figure(figsize=(8, 6))
    sentiment_counts = df['target'].value_counts().sort_index()
    plt.bar(['Negative', 'Positive'], sentiment_counts.values)
    plt.title('Distribution of Sentiments')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.savefig(os.path.join("results", "sentiment_distribution.png"))
    plt.close()
    
    # Print the counts
    print("\nSentiment Distribution:")
    print(f"Negative tweets: {sentiment_counts[0]:,}")
    print(f"Positive tweets: {sentiment_counts[1]:,}")

def analyze_tweet_lengths(df):
    """Analyze and visualize tweet lengths."""
    df['tweet_length'] = df['text'].str.len()
    
    plt.figure(figsize=(10, 6))
    plt.hist(df['tweet_length'], bins=20)
    plt.title('Distribution of Tweet Lengths')
    plt.xlabel('Tweet Length (characters)')
    plt.ylabel('Count')
    plt.savefig(os.path.join("results", "tweet_lengths.png"))
    plt.close()
    
    # Print summary statistics
    print("\nTweet Length Statistics:")
    print(f"Average length: {df['tweet_length'].mean():.1f} characters")
    print(f"Median length: {df['tweet_length'].median():.1f} characters")
    print(f"Max length: {df['tweet_length'].max():,} characters")
    print(f"Min length: {df['tweet_length'].min():,} characters")

def analyze_common_words(df, sentiment=1):
    """Analyze and print most common words for a given sentiment."""
    # Filter tweets by sentiment
    sentiment_tweets = df[df['target'] == sentiment]['text']
    
    # Split words and count them
    all_words = []
    for tweet in sentiment_tweets:
        words = str(tweet).lower().split()
        all_words.extend(words)
    
    # Get most common words
    word_counts = Counter(all_words)
    top_words = word_counts.most_common(10)
    
    # Print top words
    print(f"\nTop 10 words in {'positive' if sentiment == 1 else 'negative'} tweets:")
    for word, count in top_words:
        print(f"{word}: {count:,}")

def main():
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = load_data()
    
    # Perform analyses
    print("Analyzing sentiment distribution...")
    analyze_sentiment_distribution(df)
    
    print("Analyzing tweet lengths...")
    analyze_tweet_lengths(df)
    
    print("Analyzing common words in positive tweets...")
    analyze_common_words(df, sentiment=1)
    
    print("Analyzing common words in negative tweets...")
    analyze_common_words(df, sentiment=0)
    
    print("\nAnalysis complete! Results saved in the 'results' directory.")

if __name__ == "__main__":
    main() 