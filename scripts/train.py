import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def load_data():
    """Load the movie reviews dataset"""
    if not os.path.exists('movie_reviews.csv'):
        print("Dataset not found. Generating sample data...")
        from generate_sample_data import generate_sample_data
        df = generate_sample_data()
    else:
        df = pd.read_csv('movie_reviews.csv')
    
    return df

def preprocess_and_train():
    """Train the sentiment analysis model"""
    print("Loading data...")
    df = load_data()
    
    # Split the data
    X = df['review']
    y = df['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Vectorize the text using TF-IDF
    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train Logistic Regression model
    print("Training Logistic Regression model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_tfidf)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the model and vectorizer
    print("\nSaving model and vectorizer...")
    joblib.dump(model, 'sentiment_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    
    print("Training completed successfully!")
    print("Model saved as 'sentiment_model.pkl'")
    print("Vectorizer saved as 'tfidf_vectorizer.pkl'")

if __name__ == "__main__":
    preprocess_and_train()
