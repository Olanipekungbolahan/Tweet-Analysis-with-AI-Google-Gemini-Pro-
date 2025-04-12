import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import time
import pandas as pd
import plotly.express as px
from datetime import datetime
from pathlib import Path

# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Debug information
st.sidebar.markdown("### Debug Information")
st.sidebar.write(f"Current directory: {os.getcwd()}")
st.sidebar.write(f"Env file exists: {env_path.exists()}")

# Configure Gemini
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    st.error("""
    Please set the GOOGLE_API_KEY environment variable.
    
    Steps to fix:
    1. Create a .env file in the same directory as this script
    2. Add your API key: GOOGLE_API_KEY="your-api-key-here"
    3. Make sure the .env file is in the correct location
    """)
    st.stop()

# Verify API key format
if not api_key.startswith('AIza'):
    st.error("Invalid API key format. Google API keys typically start with 'AIza'")
    st.stop()

# Configure Gemini with error handling
try:
    genai.configure(api_key=api_key)
    # Test the API key by listing available models
    models = genai.list_models()
    st.sidebar.write("API Key Status: ✅ Valid")
    st.sidebar.write(f"Available Models: {len(models)}")
except Exception as e:
    st.error(f"Error configuring Gemini: {str(e)}")
    st.stop()

model = genai.GenerativeModel('gemini-1.0-pro',
    generation_config={
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 40,
    }
)

def predict_sentiment(text):
    """Predict sentiment using Gemini Pro."""
    try:
        prompt = f"""
        Analyze the sentiment of the following text and respond with ONLY one of these options:
        - POSITIVE
        - NEGATIVE
        - NEUTRAL
        
        Text: {text}
        """
        
        response = model.generate_content(prompt)
        sentiment = response.text.strip().upper()
        
        # Validate sentiment
        if sentiment not in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
            sentiment = "NEUTRAL"
            
        return sentiment
    except Exception as e:
        st.error(f"Error predicting sentiment: {str(e)}")
        return None

# Page config
st.set_page_config(
    page_title="Sentiment Analysis with Gemini Pro",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .sentiment-positive {
        color: #4CAF50;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #f44336;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #2196F3;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("About")
    st.markdown("""
    This application uses Google's Gemini Pro model to analyze the sentiment of text.
    
    - Enter text in the main area
    - Click 'Analyze' to get sentiment
    - View history and statistics
    """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit and Gemini Pro")

# Main content
st.title("Sentiment Analysis with Gemini Pro")
st.markdown("Analyze the sentiment of any text using advanced AI")

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Text input
text = st.text_area("Enter text to analyze:", height=150)

# Analyze button
if st.button("Analyze Sentiment"):
    if text:
        with st.spinner("Analyzing sentiment..."):
            start_time = time.time()
            sentiment = predict_sentiment(text)
            processing_time = time.time() - start_time
            
            if sentiment:
                # Display result
                st.markdown("### Result")
                sentiment_class = f"sentiment-{sentiment.lower()}"
                st.markdown(f'<p class="{sentiment_class}">Sentiment: {sentiment}</p>', unsafe_allow_html=True)
                st.markdown(f"Processing time: {processing_time:.2f} seconds")
                
                # Add to history
                st.session_state.history.append({
                    'text': text,
                    'sentiment': sentiment,
                    'timestamp': datetime.now()
                })
                
                # Display history
                st.markdown("### Analysis History")
                if st.session_state.history:
                    history_df = pd.DataFrame(st.session_state.history)
                    st.dataframe(history_df[['text', 'sentiment', 'timestamp']])
                    
                    # Sentiment distribution chart
                    sentiment_counts = history_df['sentiment'].value_counts()
                    fig = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title="Sentiment Distribution",
                        color=sentiment_counts.index,
                        color_discrete_map={
                            "POSITIVE": "#4CAF50",
                            "NEGATIVE": "#f44336",
                            "NEUTRAL": "#2196F3"
                        }
                    )
                    st.plotly_chart(fig)
    else:
        st.warning("Please enter some text to analyze") 