import os
import re
import string
import pickle
import sqlite3
import requests
import urllib.request
import zipfile
from urllib.parse import urlparse
import numpy as np
from scipy.sparse import hstack, csr_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 1. Initialize NLTK
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# 2. Define Paths First (Fixes the NameError)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '../models')
DB_PATH = os.path.join(BASE_DIR, 'domains.db')

SAFE_BROWSING_API_KEY = os.getenv("SAFE_BROWSING_API_KEY", "YOUR_ACTUAL_API_KEY_HERE")
SAFE_BROWSING_URL = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={SAFE_BROWSING_API_KEY}"

# 3. Define the Database Builder
def ensure_domain_database():
    """Builds domains.db automatically on server startup if missing."""
    if not os.path.exists(DB_PATH):
        print("[INFO] domains.db not found. Building from Tranco Top 1M list...")
        try:
            tranco_url = "https://tranco-list.eu/top-1m.csv.zip"
            zip_path = os.path.join(BASE_DIR, "temp_top1m.zip")
            csv_path = os.path.join(BASE_DIR, "top-1m.csv")

            urllib.request.urlretrieve(tranco_url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(BASE_DIR)

            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE domains (domain TEXT PRIMARY KEY);")

            batch = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) == 2:
                        batch.append((parts[1].lower(),))
                        if len(batch) >= 50000:
                            cursor.executemany("INSERT OR IGNORE INTO domains VALUES (?);", batch)
                            batch = []

            if batch:
                cursor.executemany("INSERT OR IGNORE INTO domains VALUES (?);", batch)

            conn.commit()
            conn.close()

            if os.path.exists(zip_path): os.remove(zip_path)
            if os.path.exists(csv_path): os.remove(csv_path)
            print("[SUCCESS] domains.db built successfully on instance.")
        except Exception as e:
            print(f"[WARNING] Failed to build domains.db automatically: {e}")

# 4. Execute the builder
ensure_domain_database()

# 5. Load Models
def load_model(filename):
    file_path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            print(f"[SUCCESS] Loaded {filename}")
            return pickle.load(f)
    print(f"[ERROR] Could not find {filename}")
    return None

fake_news_model = load_model('fake_news_model.pkl')
vectorizer = load_model('vectorizer.pkl')
phishing_model = load_model('phishing_model.pkl')
phishing_vectorizer = load_model('phishing_vectorizer.pkl')

def clean_text(text):
    text = str(text).lower()
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

def is_top_1m_domain(url):
    """Layer 1: Whitelist - Executes on-disk B-Tree index lookup via SQLite."""
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        parsed = urlparse(url)
        netloc = parsed.netloc.lower().split(':')[0]

        parts = netloc.split('.')
        candidates = [netloc]
        if len(parts) >= 2:
            candidates.append(f"{parts[-2]}.{parts[-1]}")
        if len(parts) >= 3:
            candidates.append(f"{parts[-3]}.{parts[-2]}.{parts[-1]}")

        if not os.path.exists(DB_PATH):
            return False

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        placeholders = ','.join('?' for _ in candidates)
        cursor.execute(f"SELECT 1 FROM domains WHERE domain IN ({placeholders}) LIMIT 1;", candidates)
        match = cursor.fetchone()
        conn.close()

        return match is not None
    except Exception:
        return False

def check_google_threat_intel(target_url):
    """Layer 2: Blacklist - Queries Google Safe Browsing API v4."""
    payload = {
        "client": {"clientId": "ai-fraud-detector", "clientVersion": "1.0.0"},
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": target_url}]
        }
    }
    try:
        response = requests.post(SAFE_BROWSING_URL, json=payload, timeout=2.5)
        if response.status_code == 200:
            data = response.json()
            if "matches" in data and len(data["matches"]) > 0:
                return "Phishing"
            return "Clean"
    except Exception as e:
        print(f"[WARNING] Threat API query error: {e}")
    return None

def predict_news(raw_text):
    if not fake_news_model or not vectorizer:
        return {"prediction": "Model Not Trained", "confidence": 0}
    try:
        cleaned = clean_text(raw_text)
        tfidf_features = vectorizer.transform([cleaned])
        stylo_features = extract_single_stylometry(raw_text)
        full_features = hstack([tfidf_features, csr_matrix(stylo_features)])
        
        pred = fake_news_model.predict(full_features)[0]
        probs = fake_news_model.predict_proba(full_features)[0]
        result = "Real" if pred == 1 else "Fake"
        confidence = round(max(probs) * 100, 2)
        return {"prediction": result, "confidence": confidence}
    except Exception:
        return {"prediction": "Processing Error", "confidence": 0}

def predict_phishing(url):
    if not phishing_model or not phishing_vectorizer:
        return {"prediction": "Model Not Trained", "confidence": 0}
    try:
        # --- LAYER 1: SQLite Domain Whitelist ---
        if is_top_1m_domain(url):
            return {"prediction": "Safe", "confidence": 99.80}

        # --- LAYER 2: Google Safe Browsing Blacklist ---
        api_verdict = check_google_threat_intel(url)
        if api_verdict == "Phishing":
            return {"prediction": "Phishing", "confidence": 99.99}

        # --- LAYER 3: Machine Learning Fallback ---
        vectorized_url = phishing_vectorizer.transform([url])
        pred = phishing_model.predict(vectorized_url)[0]
        probs = phishing_model.predict_proba(vectorized_url)[0]
        confidence = round(max(probs) * 100, 2)

        # Give a slight boost if Google confirmed it wasn't on a blacklist, but ML thinks it's Safe
        if api_verdict == "Clean" and pred == 0:
            confidence = max(confidence, 96.00)

        result = "Phishing" if pred == 1 else "Safe"
        return {"prediction": result, "confidence": confidence}
    except Exception:
        return {"prediction": "Processing Error", "confidence": 0}