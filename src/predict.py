import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import time
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

def load_data():
    """Load the preprocessed dataset."""
    data_path = os.path.join("data", "preprocessed_data.csv")
    return pd.read_csv(data_path)

def setup_gemini():
    """Set up the Gemini Pro model."""
    # Load environment variables
    load_dotenv()
    
    # Get API key from environment variable
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("Please set the GOOGLE_API_KEY environment variable")
    
    # Configure the model
    genai.configure(api_key=api_key)
    
    # Create the model with safety settings
    model = genai.GenerativeModel('gemini-pro',
        generation_config={
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40,
        },
        safety_settings=[
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
    )
    return model

def predict_sentiment(model, text):
    """Predict sentiment using Gemini Pro with an improved prompt."""
    prompt = f"""
    Analyze the sentiment of the following tweet. Consider the following aspects:
    1. Overall tone and emotion
    2. Use of positive/negative words
    3. Context and implications
    4. Sarcasm or irony if present
    
    Tweet: "{text}"
    
    Respond with ONLY one of these exact words:
    - "positive" if the sentiment is positive
    - "negative" if the sentiment is negative
    - "neutral" if the sentiment is neutral
    
    Do not include any explanation or additional text.
    """
    
    try:
        response = model.generate_content(prompt)
        sentiment = response.text.strip().lower()
        
        # Map sentiment to numeric value
        if 'positive' in sentiment:
            return 1
        elif 'negative' in sentiment:
            return 0
        else:
            return 0.5  # Neutral
    except Exception as e:
        print(f"Error predicting sentiment: {e}")
        return None

def evaluate_predictions(df, predictions):
    """Evaluate the predictions with multiple metrics."""
    # Calculate metrics
    accuracy = sum(p == t for p, t in zip(predictions, df['target'])) / len(df)
    precision = precision_score(df['target'], predictions, average='binary')
    recall = recall_score(df['target'], predictions, average='binary')
    f1 = f1_score(df['target'], predictions, average='binary')
    
    # Create confusion matrix
    cm = confusion_matrix(df['target'], predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join("results", "confusion_matrix.png"))
    plt.close()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def save_results(df, predictions, metrics):
    """Save prediction results and metrics."""
    # Save predictions
    results_df = pd.DataFrame({
        'text': df['text'],
        'true_sentiment': df['target'],
        'predicted_sentiment': predictions
    })
    results_path = os.path.join("results", "predictions.csv")
    results_df.to_csv(results_path, index=False)
    
    # Save metrics
    metrics_path = os.path.join("results", "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Results saved to {results_path}")
    print(f"Metrics saved to {metrics_path}")

def test_new_tweet(model, tweet):
    """Test the predictor on a new tweet."""
    print(f"\nAnalyzing tweet: {tweet}")
    prediction = predict_sentiment(model, tweet)
    sentiment = "Positive" if prediction == 1 else "Negative" if prediction == 0 else "Neutral"
    print(f"Predicted sentiment: {sentiment}")

def main():
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = load_data()
    
    # Set up Gemini Pro
    print("Setting up Gemini Pro...")
    model = setup_gemini()
    
    # Make predictions
    print("Making predictions...")
    predictions = []
    for text in tqdm(df['text']):
        prediction = predict_sentiment(model, text)
        predictions.append(prediction)
        # Add a small delay to avoid rate limiting
        time.sleep(0.5)
    
    # Evaluate predictions
    print("\nEvaluating predictions...")
    metrics = evaluate_predictions(df, predictions)
    
    # Print metrics
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall: {metrics['recall']:.2%}")
    print(f"F1 Score: {metrics['f1_score']:.2%}")
    
    # Save results
    save_results(df, predictions, metrics)
    
    # Test on new tweets
    print("\nTesting on new tweets...")
    test_tweets = [
        "I love this new feature! It's amazing!",
        "This is the worst experience I've ever had.",
        "The weather is nice today.",
        "I'm not sure how I feel about this."
    ]
    
    for tweet in test_tweets:
        test_new_tweet(model, tweet)

if __name__ == "__main__":
    main() 