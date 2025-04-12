# Sentiment Analysis with Gemini Pro

This project provides a sentiment analysis service using Google's Gemini Pro model. It includes a web interface and API endpoint for analyzing text sentiment.

## Features

- Sentiment analysis using Gemini Pro
- Web interface for easy interaction
- REST API endpoint for programmatic access
- Prometheus metrics for monitoring
- Heroku deployment ready

## Project Structure

```
.
├── app.py                 # Flask application
├── Procfile              # Heroku process file
├── requirements.txt      # Python dependencies
├── runtime.txt          # Python version
├── templates/           # HTML templates
│   └── index.html      # Web interface
├── data/               # Data directory
└── src/               # Source code
    ├── predict.py     # Prediction module
    └── analyze.py     # Analysis module
```

## Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd sentiment-analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a .env file with your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

5. Run the application:
```bash
python app.py
```

## Heroku Deployment

1. Install the Heroku CLI:
```bash
curl https://cli-assets.heroku.com/install.sh | sh
```

2. Login to Heroku:
```bash
heroku login
```

3. Create a new Heroku app:
```bash
heroku create your-app-name
```

4. Set environment variables:
```bash
heroku config:set GOOGLE_API_KEY=your_api_key_here
```

5. Deploy to Heroku:
```bash
git push heroku main
```

6. Open the application:
```bash
heroku open
```

## API Usage

### Endpoint: `/predict`
- Method: POST
- Content-Type: application/json
- Request body:
```json
{
    "text": "Your text to analyze"
}
```
- Response:
```json
{
    "sentiment": "positive|negative|neutral",
    "confidence": 1.0,
    "processing_time": 0.123
}
```

## Monitoring

The application exposes Prometheus metrics at `/metrics` endpoint. You can use these metrics to monitor:
- Total number of predictions
- Prediction latency
- Error rates

## License

This project is licensed under the MIT License - see the LICENSE file for details. 