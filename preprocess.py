import pandas as pd
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

nltk.download('punkt')
ps = PorterStemmer()

# Text Preprocessing
def process_email(mail):
    mail = mail.lower()
    mail = re.sub(r'<[^<>]+>', ' ', mail)  # Remove HTML tags
    mail = re.sub(r'\d+', 'number', mail)  # Replace numbers
    mail = re.sub(r'(http|https)://\S+', 'httpaddr', mail)  # Replace URLs
    mail = re.sub(r'\S+@\S+', 'emailaddr', mail)  # Replace email addresses
    mail = re.sub(r'[$]+', 'dollar', mail)  # Replace dollar signs
    mail = re.sub(r'[^a-zA-Z]', ' ', mail)  # Remove non-alphabetic characters
    words = word_tokenize(mail)
    return ' '.join(ps.stem(word) for word in words if len(word) > 1)

# Load Dataset & Preprocess
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, encoding='latin-1').iloc[:, :2]
    df.columns = ['label', 'text']
    
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Encode labels
    df['processed_text'] = df['text'].apply(process_email)

    return df

# Vectorization
def vectorize_data(df, max_features=2000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df['processed_text']).toarray()
    y = df['label']
    
    joblib.dump(vectorizer, 'vectorizer.pkl')  # Save vectorizer
    return X, y

# Split Data
def split_data(X, y):
    return train_test_split(X, y, train_size=0.8, random_state=42)
