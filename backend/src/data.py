import pandas as pd
from preprocessing import preprocess_text  # Import fungsi preprocess_text 

def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        # reprocessing the 'review' column
        df['preprocessed_review'] = df['review'].apply(preprocess_text)
        return df
    except FileNotFoundError:
        print("File dataset tidak ditemukan.")
        return None

if __name__ == "__main__":
    file_path = "data/dataset.csv"  
    dataset = load_dataset(file_path)
    if dataset is not None:
        print("Dataset berhasil dimuat.")
    else:
        print("Gagal memuat dataset.")
