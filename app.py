from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from dotenv import load_dotenv
import os
import logging
from prometheus_client import Counter, Histogram, make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter('sentiment_predictions_total', 'Total number of sentiment predictions')
PREDICTION_LATENCY = Histogram('sentiment_prediction_latency_seconds', 'Time spent processing predictions')

# Load environment variables
load_dotenv()

# Configure Gemini
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.0-pro',
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
        }
    ]
)

def predict_sentiment(text):
    """Predict sentiment using Gemini Pro."""
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
        
        if 'positive' in sentiment:
            return 1
        elif 'negative' in sentiment:
            return 0
        else:
            return 0.5
    except Exception as e:
        logger.error(f"Error predicting sentiment: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@PREDICTION_LATENCY.time()
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        start_time = time.time()
        sentiment = predict_sentiment(text)
        prediction_time = time.time() - start_time
        
        if sentiment is None:
            return jsonify({'error': 'Failed to predict sentiment'}), 500
        
        PREDICTION_COUNTER.inc()
        
        result = {
            'sentiment': 'positive' if sentiment == 1 else 'negative' if sentiment == 0 else 'neutral',
            'confidence': 1.0,  # Gemini doesn't provide confidence scores
            'processing_time': prediction_time
        }
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({'error': str(e)}), 500

# Add prometheus metrics endpoint
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 