import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    """
    Preprocessing teks:
    - Mengubah teks menjadi huruf kecil
    - Menghapus karakter khusus, angka, dan tanda baca
    - Tokenisasi teks
    - Menghapus stop words
    - Lematisasi kata
    
    Parameters:
        text (str): Teks yang akan diproses.
    
    Returns:
        str: Teks yang sudah diproses.
    """
    # Lower Teks
    text = text.lower()
    
    # Menghapus karakter khusus, angka, dan tanda baca
    text = re.sub(r'\d+', '', text)  # Menghapus angka
    text = text.translate(str.maketrans('', '', string.punctuation))  # Menghapus tanda baca
    
    # Tokenisasi teks
    tokens = word_tokenize(text)
    
    # Menghapus stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Lematisasi word
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    # reuse words that have been lemmatized into text
    processed_text = ' '.join(lemmatized_text)
    
    return processed_text

if __name__ == "__main__":
    # loading dataset
    file_path = "data/dataset.csv"
    dataset = pd.read_csv(file_path)
    
    # Check and deal with missing values in the 'review' column
    dataset['review'].fillna('', inplace=True)
    
    # Proses preprocessing teks
    dataset['preprocessed_review'] = dataset['review'].apply(preprocess_text)
    
    # Save the processed dataset into a CSV file
    preprocessed_file_path = "data/preprocessed_data.csv"
    dataset.to_csv(preprocessed_file_path, index=False)
    
    print("Hasil preprocessing telah disimpan di:", preprocessed_file_path)
