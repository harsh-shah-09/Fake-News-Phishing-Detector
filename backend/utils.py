import pickle
import os
import re
import string
import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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
phishing_model = load_model('phishing_model.pkl')
phishing_vectorizer = load_model('phishing_vectorizer.pkl')

def clean_text(text):
    """Must match the training cleaner exactly."""
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('\w*\d\w*', '', text)
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

def predict_news(text):
    if not fake_news_model or not vectorizer:
        return {"prediction": "Model Not Trained", "confidence": 0}
    try:
        # CRITICAL FIX: Clean the text before vectorizing
        cleaned_input = clean_text(text)
        vectorized_text = vectorizer.transform([cleaned_input])
        
        prediction_value = fake_news_model.predict(vectorized_text)[0]
        probabilities = fake_news_model.predict_proba(vectorized_text)[0]
        confidence = round(max(probabilities) * 100, 2)
        
        result = "Fake" if prediction_value == 0 else "Real"
        return {"prediction": result, "confidence": confidence}
    except Exception as e:
        return {"prediction": "Processing Error", "confidence": 0}

def predict_phishing(url):
    if not phishing_model or not phishing_vectorizer:
        return {"prediction": "Model Not Trained", "confidence": 0}
    try:
        # Transform the incoming URL using Lexical Analysis
        vectorized_url = phishing_vectorizer.transform([url])
        
        prediction_value = phishing_model.predict(vectorized_url)[0]
        probabilities = phishing_model.predict_proba(vectorized_url)[0]
        confidence = round(max(probabilities) * 100, 2)
        
        result = "Phishing" if prediction_value == 1 else "Safe"
        return {"prediction": result, "confidence": confidence}
    except Exception as e:
        return {"prediction": "Processing Error", "confidence": 0}