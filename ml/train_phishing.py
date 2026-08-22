import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def load_and_prepare_data():
    print("[INFO] Loading Phishing Dataset...")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_PATH = os.path.join(BASE_DIR, r'C:\Users\admin\Desktop\Fake-News-Phishing-Detector\datasets\phishing_site_urls.csv')
    
    try:
        df = pd.read_csv(DATASET_PATH)
        # Logistic Regression is highly efficient, so 150k rows is perfect!
        df = df.sample(n=150000, random_state=42).reset_index(drop=True)
    except FileNotFoundError:
        print(f"[ERROR] Dataset not found at: {DATASET_PATH}")
        return None, None
        
    df['Label'] = df['Label'].map({'good': 0, 'bad': 1})
    return df['URL'], df['Label']

def train_model():
    X_raw, Y = load_and_prepare_data()
    if X_raw is None: return
    
    print("[INFO] Vectorizing URLs (Character-Level TF-IDF)...")
    # Analyzes sub-word character chunks (e.g., catching "paypa1")
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=5000)
    X = vectorizer.fit_transform(X_raw)
    
    print("[INFO] Splitting dataset into Training and Testing sets...")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    print("[INFO] Training Logistic Regression Model...")
    model = LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, Y_train)
    
    print("\n[INFO] Evaluating Model Performance:")
    predictions = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(Y_test, predictions):.2%}")
    print("\nClassification Report:\n", classification_report(Y_test, predictions))
    
    print("[INFO] Saving Model and Vectorizer to disk...")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(BASE_DIR, '../models/phishing_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(BASE_DIR, '../models/phishing_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    print("[SUCCESS] Phishing model training complete and saved.")

if __name__ == '__main__':
    train_model()