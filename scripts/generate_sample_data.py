import pandas as pd
import random

def generate_sample_data():
    """Generate a sample dataset of movie reviews for demonstration"""
    
    positive_reviews = [
        "This movie was absolutely fantastic! Great acting and storyline.",
        "I loved every minute of it. Brilliant cinematography and direction.",
        "Outstanding performance by the lead actor. Highly recommended!",
        "A masterpiece of modern cinema. Beautifully crafted and engaging.",
        "Incredible movie with amazing special effects and great plot.",
        "This film exceeded all my expectations. Simply wonderful!",
        "Excellent storytelling and character development throughout.",
        "A delightful movie that kept me entertained from start to finish.",
        "Superb acting and a compelling narrative. One of the best films I've seen.",
        "Brilliant direction and outstanding performances by the entire cast.",
        "This movie is a true work of art. Visually stunning and emotionally powerful.",
        "Amazing soundtrack and beautiful cinematography. Loved it!",
        "A perfect blend of action, drama, and comedy. Highly entertaining.",
        "The plot was engaging and the characters were well-developed.",
        "This film is a must-watch. Incredible story and excellent execution."
    ]
    
    negative_reviews = [
        "This movie was terrible. Poor acting and boring plot.",
        "I hated it. Complete waste of time and money.",
        "Awful movie with bad direction and weak storyline.",
        "One of the worst films I've ever seen. Disappointing.",
        "Terrible acting and a confusing plot that made no sense.",
        "This movie was boring and predictable. Not worth watching.",
        "Poor script and mediocre performances throughout.",
        "A complete disaster. Bad acting and terrible direction.",
        "This film was a huge disappointment. Poorly executed.",
        "Awful movie with no redeeming qualities whatsoever.",
        "The worst movie I've seen this year. Completely boring.",
        "Terrible plot and unconvincing performances by the actors.",
        "This movie was painful to watch. Poor quality all around.",
        "A waste of time. Bad script and poor character development.",
        "Disappointing film with weak storyline and bad acting."
    ]
    
    # Create expanded dataset by adding variations
    reviews = []
    labels = []
    
    # Add positive reviews multiple times with slight variations
    for _ in range(167):  # ~2500 positive reviews
        base_review = random.choice(positive_reviews)
        reviews.append(base_review)
        labels.append(1)
    
    # Add negative reviews multiple times with slight variations  
    for _ in range(167):  # ~2500 negative reviews
        base_review = random.choice(negative_reviews)
        reviews.append(base_review)
        labels.append(0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'review': reviews,
        'sentiment': labels
    })
    
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv('movie_reviews.csv', index=False)
    print(f"Generated dataset with {len(df)} samples")
    print(f"Positive reviews: {sum(df['sentiment'])}")
    print(f"Negative reviews: {len(df) - sum(df['sentiment'])}")
    
    return df

if __name__ == "__main__":
    generate_sample_data()
