import pickle
import os
import re
import string
import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from scipy.sparse import hstack, csr_matrix
from urllib.parse import urlparse

# List of universally verified top-level authoritative root domains
TRUSTED_ROOT_DOMAINS = {
    'google.com', 'google.co.in', 'youtube.com', 'github.com',
    'microsoft.com', 'apple.com', 'amazon.com', 'linkedin.com',
    'wikipedia.org', 'cloudflare.com', 'mozilla.org'
}
def is_trusted_domain(url):
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        parsed = urlparse(url)
        netloc = parsed.netloc.lower().split(':')[0]
        
        # Check direct domain match or parent root domain match
        for trusted in TRUSTED_ROOT_DOMAINS:
            if netloc == trusted or netloc.endswith('.' + trusted):
                return True
        return False
    except Exception:
        return False

# Initialize NLTK safely for the server
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '../models')

def load_model(filename):
    file_path = os.path.join(MODELS_DIR, filename)
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        print(f"[SUCCESS] Loaded {filename}")
        return model
    except FileNotFoundError:
        return None

fake_news_model = load_model('fake_news_model.pkl')
vectorizer = load_model('vectorizer.pkl')
phishing_model = load_model('phishing_model.pkl')
phishing_vectorizer = load_model('phishing_vectorizer.pkl')

def clean_text(text):
    """Must match the training cleaner exactly."""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

def extract_single_stylometry(text_str):
    total_chars = max(len(text_str), 1)
    words = text_str.split()
    total_words = max(len(words), 1)
    
    cap_count = sum(1 for c in text_str if c.isupper())
    excl_count = text_str.count('!')
    ques_count = text_str.count('?')
    punct_count = sum(1 for c in text_str if c in string.punctuation)
    avg_word_len = sum(len(w) for w in words) / total_words
    lexical_diversity = len(set(words)) / total_words
    
    return np.array([[
        cap_count / total_chars,
        excl_count / total_words,
        ques_count / total_words,
        punct_count / total_chars,
        avg_word_len,
        lexical_diversity
    ]])

def predict_news(raw_text):
    if not fake_news_model or not vectorizer:
        return {"prediction": "Model Not Trained", "confidence": 0}
    try:
        cleaned = clean_text(raw_text)
        tfidf_features = vectorizer.transform([cleaned])
        stylo_features = extract_single_stylometry(raw_text)
        
        # Fuse input data
        full_features = hstack([tfidf_features, csr_matrix(stylo_features)])
        
        pred = fake_news_model.predict(full_features)[0]
        probs = fake_news_model.predict_proba(full_features)[0]
        
        result = "Real" if pred == 1 else "Fake"
        confidence = round(max(probs) * 100, 2)
        return {"prediction": result, "confidence": confidence}
    except Exception as e:
        return {"prediction": "Processing Error", "confidence": 0}

def predict_phishing(url):
    if not phishing_model or not phishing_vectorizer:
        return {"prediction": "Model Not Trained", "confidence": 0}
    try:
        # Step 1: Check Domain Authority Guardrail
        if is_trusted_domain(url):
            return {
                "prediction": "Safe",
                "confidence": 98.50
            }

        # Step 2: Machine Learning Lexical Evaluation
        vectorized_url = phishing_vectorizer.transform([url])
        prediction_value = phishing_model.predict(vectorized_url)[0]
        probabilities = phishing_model.predict_proba(vectorized_url)[0]
        confidence = round(max(probabilities) * 100, 2)

        result = "Phishing" if prediction_value == 1 else "Safe"
        return {"prediction": result, "confidence": confidence}
    except Exception as e:
        return {"prediction": "Processing Error", "confidence": 0}