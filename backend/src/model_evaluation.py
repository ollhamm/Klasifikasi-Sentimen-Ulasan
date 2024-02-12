import joblib
import pandas as pd
from sklearn.metrics import classification_report

if __name__ == "__main__":
    # Load model that have been try
    model = joblib.load('models/sentiment_classifier.pkl')

    # Load dataset uji
    test_data = pd.read_csv('data/test_data.csv')

    # Replace the NaN value with an empty string in the 'review' column
    test_data['review'].fillna('', inplace=True)

    # perform prediction on test data
    X_test = test_data['review']
    y_test = test_data['score']
    y_pred = model.predict(X_test)

    # Display kalsifikasi report
    print(classification_report(y_test, y_pred))
