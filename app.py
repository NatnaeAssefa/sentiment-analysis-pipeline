from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Global variables to store model and vectorizer
model = None
vectorizer = None

def load_model():
    """Load the trained model and vectorizer"""
    global model, vectorizer
    try:
        model = joblib.load('sentiment_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        print("Model and vectorizer loaded successfully!")
    except FileNotFoundError:
        print("Error: Model files not found. Please run 'python scripts/train.py' first.")
        return False
    return True

def predict_sentiment(text):
    """Predict sentiment for a given text"""
    if model is None or vectorizer is None:
        return None, None
    
    # Vectorize the input text
    text_tfidf = vectorizer.transform([text])
    
    # Get prediction and probability
    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]
    
    # Get confidence score
    confidence = max(probabilities)
    
    # Convert prediction to label
    sentiment = "positive" if prediction == 1 else "negative"
    
    return sentiment, confidence

@app.route('/')
def home():
    """Home endpoint with API information"""
    return jsonify({
        "message": "Sentiment Analysis API",
        "endpoints": {
            "/predict": "POST - Predict sentiment of text",
            "/health": "GET - Check API health"
        },
        "example": {
            "url": "/predict",
            "method": "POST",
            "body": {"text": "I loved this movie!"}
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None and vectorizer is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict sentiment endpoint"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "error": "Please provide 'text' field in JSON body"
            }), 400
        
        text = data['text']
        
        if not text.strip():
            return jsonify({
                "error": "Text cannot be empty"
            }), 400
        
        # Make prediction
        sentiment, confidence = predict_sentiment(text)
        
        if sentiment is None:
            return jsonify({
                "error": "Model not loaded properly"
            }), 500
        
        return jsonify({
            "text": text,
            "sentiment": sentiment,
            "confidence": round(confidence, 4)
        })
    
    except Exception as e:
        return jsonify({
            "error": f"An error occurred: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        print("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Exiting...")
