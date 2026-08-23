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
from tranco import Tranco


print("[INFO] Booting Domain Authority Guardrail...")
try:
    # This automatically downloads and caches the Top 1 Million list on startup
    t = Tranco(cache=True, cache_dir='.tranco')
    tranco_list = t.list()
    TRANCO_AVAILABLE = True
    print(f"[SUCCESS] Tranco Top 1 Million loaded. List ID: {tranco_list.list_id}")
except Exception as e:
    print("[WARNING] Tranco API failed. Falling back to emergency whitelist.")
    TRANCO_AVAILABLE = False
    EMERGENCY_WHITELIST = {'google.com', 'google.co.in', 'youtube.com', 'github.com','microsoft.com', 'apple.com', 'amazon.com', 'linkedin.com','wikipedia.org', 'cloudflare.com', 'mozilla.org','indianexpress.com', 'thehindu.com', 'timesofindia.indiatimes.com','ndtv.com', 'bbc.com', 'bbc.co.uk', 'cnn.com', 'reuters.com','nytimes.com', 'wsj.com', 'aljazeera.com', 'bloomberg.com','twitter.com', 'x.com', 'facebook.com', 'instagram.com', 'whatsapp.com', 'telegram.org', 'reddit.com', 'quora.com', 'medium.com', 'stackexchange.com', 'stackoverflow.com','w3schools.com', 'geeksforgeeks.org', 'coursera.org', 'edx.org', 'khanacademy.org', 'udemy.com', 'pluralsight.com','linkedin.com', 'indeed.com', 'glassdoor.com', 'monster.com', 'naukri.com', 'timesjobs.com', 'shine.com', 'freshersworld.com', 'hackernews.com', 'dev.to', 'hashnode.com',}


def is_trusted_domain(url):
    """
    Extracts the root domain and dynamically queries the Tranco Top 1M list.
    """
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        parsed = urlparse(url)
        netloc = parsed.netloc.lower().split(':')[0]
        
        # Create variations to handle subdomains (gemini.google.com -> google.com)
        parts = netloc.split('.')
        domains_to_check = [netloc]
        if len(parts) >= 2:
            domains_to_check.append(f"{parts[-2]}.{parts[-1]}") # Root (google.com)
        if len(parts) >= 3:
            domains_to_check.append(f"{parts[-3]}.{parts[-2]}.{parts[-1]}") # Country code (google.co.in)
        
        # 1. Query the Global API List
        if TRANCO_AVAILABLE:
            for d in domains_to_check:
                # If rank() returns anything other than -1, the domain is in the Top 1M
                if tranco_list.rank(d) != -1:
                    return True
            return False
            
        # 2. Emergency Fallback
        else:
            return any(d in EMERGENCY_WHITELIST for d in domains_to_check)
            
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