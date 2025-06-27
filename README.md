# Sentiment Analysis ML Pipeline

A complete machine learning pipeline for sentiment analysis of movie reviews using scikit-learn, TF-IDF vectorization, and Logistic Regression.

## Features

- **Text Classification**: Classify movie reviews as positive or negative
- **TF-IDF Vectorization**: Convert text to numerical features
- **Logistic Regression**: Simple yet effective classification model
- **Command-Line Interface**: Easy-to-use prediction script
- **Web API**: Flask-based REST API for predictions
- **Sample Dataset**: Includes sample movie reviews for demonstration

## Installation

1. Clone or download this project
2. Install the required dependencies:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Usage

### 1. Generate Sample Data and Train the Model

First, generate the sample dataset and train the model:

\`\`\`bash
python scripts/generate_sample_data.py
python scripts/train.py
\`\`\`

This will:
- Generate a sample dataset of movie reviews (\`movie_reviews.csv\`)
- Train a Logistic Regression model with TF-IDF features
- Save the trained model (\`sentiment_model.pkl\`) and vectorizer (\`tfidf_vectorizer.pkl\`)
- Display training accuracy and evaluation metrics

### 2. Command-Line Predictions

Use the command-line script to predict sentiment:

\`\`\`bash
python predict.py "I loved this movie!"
python predict.py "This movie was terrible and boring."
\`\`\`

Output format:
\`\`\`
Review: I loved this movie!
Sentiment: positive
Confidence: 0.8542
\`\`\`

### 3. Web API

Start the Flask web server:

\`\`\`bash
python app.py
\`\`\`

The API will be available at \`http://localhost:5000\`

#### API Endpoints

- **GET /**: API information and usage examples
- **GET /health**: Health check endpoint
- **POST /predict**: Predict sentiment of text

#### Example API Usage

\`\`\`bash
# Using curl
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I loved this movie!"}'

# Response
{
  "text": "I loved this movie!",
  "sentiment": "positive",
  "confidence": 0.8542
}
\`\`\`

## Project Structure

\`\`\`
sentiment-analysis-pipeline/
├── scripts/
│   ├── generate_sample_data.py  # Generate sample dataset
│   └── train.py                 # Train the model
├── predict.py                   # Command-line prediction script
├── app.py                       # Flask web API
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── movie_reviews.csv           # Generated dataset (after running scripts)
├── sentiment_model.pkl         # Trained model (after training)
└── tfidf_vectorizer.pkl        # TF-IDF vectorizer (after training)
\`\`\`

## Model Details

- **Algorithm**: Logistic Regression
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Features**: Up to 5,000 most important words and bigrams
- **Preprocessing**: English stop words removal, minimum document frequency filtering
- **Training/Test Split**: 80/20

## Performance

The model typically achieves:
- **Accuracy**: ~85-90% on the test set
- **Precision/Recall**: Balanced performance for both positive and negative classes

## Customization

### Using Your Own Dataset

Replace the sample data generation with your own dataset:

1. Create a CSV file with columns: \`review\` (text) and \`sentiment\` (0 for negative, 1 for positive)
2. Save it as \`movie_reviews.csv\`
3. Run the training script: \`python scripts/train.py\`

### Model Parameters

You can modify the model parameters in \`scripts/train.py\`:

\`\`\`python
# TF-IDF parameters
vectorizer = TfidfVectorizer(
    max_features=5000,      # Maximum number of features
    stop_words='english',   # Remove English stop words
    ngram_range=(1, 2),     # Use unigrams and bigrams
    min_df=2               # Minimum document frequency
)

# Logistic Regression parameters
model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    C=1.0                  # Regularization strength
)
\`\`\`

## Dependencies

- **scikit-learn**: Machine learning library
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **flask**: Web framework for API
- **joblib**: Model serialization

## License

This project is open source and available under the MIT License.
\`\`\`
