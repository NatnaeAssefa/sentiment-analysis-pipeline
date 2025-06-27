import sys
import joblib
import numpy as np
import os

def load_model():
    """Load the trained model and vectorizer"""
    try:
        model = joblib.load('sentiment_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        print("Error: Model files not found. Please run 'python scripts/train.py' first.")
        sys.exit(1)

def predict_sentiment(text, model, vectorizer):
    """Predict sentiment for a given text"""
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

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py \"Your review text here\"")
        print("Example: python predict.py \"I loved this movie!\"")
        sys.exit(1)
    
    review_text = sys.argv[1]
    
    # Load model
    model, vectorizer = load_model()
    
    # Make prediction
    sentiment, confidence = predict_sentiment(review_text, model, vectorizer)
    
    # Print results
    print(f"Review: {review_text}")
    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()
