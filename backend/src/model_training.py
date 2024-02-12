import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

if __name__ == "__main__":
    # Load dataset that has been processed
    data = pd.read_csv('data/preprocessed_data.csv')

    # Replace the NaN value with an empty string in the 'preprocessed_text' column
    data['preprocessed_review'].fillna('', inplace=True)

    # Split data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(data['preprocessed_review'], data['score'], test_size=0.2, random_state=42)

    # Building a pipeline for text vectorization and classification model generation
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LinearSVC())
    ])
 
    # Try model
    pipeline.fit(X_train, y_train)

    # Predict and try
    y_pred = pipeline.predict(X_test)

    # Display the classification report
    print(classification_report(y_test, y_pred))

    # Create a 'models' directory if it doesn't already exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Save model to file
    joblib.dump(pipeline, 'models/sentiment_classifier.pkl')
